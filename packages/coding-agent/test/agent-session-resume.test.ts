import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, type AssistantMessageEvent, EventStream, getModel } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader } from "./utilities.ts";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createAssistantMessage(text: string, overrides?: Partial<AssistantMessage>): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
		...overrides,
	};
}

describe("AgentSession resume", () => {
	let session: AgentSession;
	let tempDir: string;

	beforeEach(async () => {
		tempDir = join(tmpdir(), `pi-resume-test-${Date.now()}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(() => {
		if (session) {
			session.dispose();
		}
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	/**
	 * Builds a session whose first `failCount` stream calls emit an error/aborted
	 * message, and subsequent calls succeed. `retryEnabled` controls whether
	 * auto-retry fires (we want it OFF for most resume tests so the failed turn
	 * is left at the leaf for resume() to pick up).
	 */
	async function createSession(options?: {
		failCount?: number;
		maxRetries?: number;
		retryEnabled?: boolean;
		failStopReason?: "error" | "aborted";
		failErrorMessage?: string;
		streamDelayMs?: number;
	}) {
		const failCount = options?.failCount ?? 1;
		const maxRetries = options?.maxRetries ?? 0;
		const retryEnabled = options?.retryEnabled ?? false;
		const failStopReason = options?.failStopReason ?? "error";
		const failErrorMessage = options?.failErrorMessage ?? "Connection error.";
		const streamDelayMs = options?.streamDelayMs ?? 0;
		let callCount = 0;

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFunction: () => {
				callCount++;
				const stream = new MockAssistantStream();
				const emit = () => {
					if (callCount <= failCount) {
						const msg = createAssistantMessage("", {
							stopReason: failStopReason,
							errorMessage: failStopReason === "error" ? failErrorMessage : undefined,
						});
						stream.push({ type: "start", partial: msg });
						stream.push({
							type: failStopReason === "aborted" ? "done" : "error",
							reason: failStopReason,
							...(failStopReason === "aborted" ? { message: msg } : { error: msg }),
						});
					} else {
						const msg = createAssistantMessage("Success after resume");
						stream.push({ type: "start", partial: msg });
						stream.push({ type: "done", reason: "stop", message: msg });
					}
				};
				if (streamDelayMs > 0) {
					setTimeout(emit, streamDelayMs);
				} else {
					queueMicrotask(emit);
				}
				return stream;
			},
		});

		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
		settingsManager.applyOverrides({
			retry: { enabled: retryEnabled, maxRetries, baseDelayMs: 1 },
		});

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader(),
		});

		return { session, getCallCount: () => callCount };
	}

	it("resumes after a retryable error when auto-retry is disabled", async () => {
		const created = await createSession({ failCount: 1, retryEnabled: false });
		await created.session.prompt("Test");

		// Auto-retry off → only the failing call ran; leaf is the error message.
		expect(created.getCallCount()).toBe(1);

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });

		// resume() popped the failed turn and re-ran; provider saw a second call.
		expect(created.getCallCount()).toBe(2);
		expect(created.session.isIdle).toBe(true);
	});

	it("resumes after an aborted turn (the auto-retry gap)", async () => {
		const created = await createSession({
			failCount: 1,
			retryEnabled: true,
			maxRetries: 3,
			failStopReason: "aborted",
		});
		await created.session.prompt("Test");

		// Aborted stopReason is NOT retryable (retry.ts:99 requires stopReason==="error"),
		// so auto-retry never fires even with retry enabled.
		expect(created.getCallCount()).toBe(1);

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });
		expect(created.getCallCount()).toBe(2);
	});

	it("refuses when agent is not idle", async () => {
		// failCount=0 with a slow success stream: prompt() is still streaming when
		// we call resume(), so the not_idle guard must fire.
		const created = await createSession({
			failCount: 0,
			retryEnabled: false,
			streamDelayMs: 60,
		});
		const promptPromise = created.session.prompt("Test");
		// Let the stream actually start.
		await new Promise((r) => setTimeout(r, 10));
		const result = await created.session.resume();
		expect(result).toEqual({ resumed: false, reason: "not_idle" });
		await promptPromise;
	});

	it("refuses when the last turn was not failed or aborted", async () => {
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Test"); // succeeds, leaf is stopReason:"stop"

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: false, reason: "no_failed_turn" });
	});

	it("does not consume the auto-retry budget", async () => {
		const created = await createSession({ failCount: 1, retryEnabled: false });
		await created.session.prompt("Test");
		expect(created.session.retryAttempt).toBe(0);

		await created.session.resume();
		// _retryAttempt stays 0 — manual resume does not touch the auto-retry counter.
		expect(created.session.retryAttempt).toBe(0);
	});

	it("pops the failed entry from agent state but keeps it in the session file", async () => {
		const created = await createSession({ failCount: 1, retryEnabled: false });
		await created.session.prompt("Test");

		// Before resume: agent state ends with the failed assistant message.
		const stateBefore = created.session.agent.state.messages;
		const tailBefore = stateBefore[stateBefore.length - 1];
		expect(tailBefore.role).toBe("assistant");
		expect((tailBefore as AssistantMessage).stopReason).toBe("error");

		await created.session.resume();

		// After resume: agent state ends with the SUCCESS message, not the error.
		const stateAfter = created.session.agent.state.messages;
		const tailAfter = stateAfter[stateAfter.length - 1] as AssistantMessage;
		expect(tailAfter.stopReason).toBe("stop");

		// Session file still contains the error entry (parentId chain preserved).
		const entries = created.session.sessionManager.getEntries();
		const errorEntries = entries.filter(
			(e) => e.type === "message" && (e.message as AssistantMessage).stopReason === "error",
		);
		expect(errorEntries.length).toBe(1);
	});

	it("leaves the session ready for a normal follow-up prompt", async () => {
		// After resume() succeeds, the agent must be idle with retryAttempt reset,
		// so a subsequent prompt works without "Agent is already processing" errors.
		const created = await createSession({ failCount: 1, retryEnabled: false });
		await created.session.prompt("Test");
		expect(created.getCallCount()).toBe(1);

		await created.session.resume();
		expect(created.getCallCount()).toBe(2);
		expect(created.session.isIdle).toBe(true);
		expect(created.session.retryAttempt).toBe(0);

		// A brand-new prompt must go through cleanly.
		await created.session.prompt("Follow-up");
		expect(created.getCallCount()).toBe(3);
		expect(created.session.isIdle).toBe(true);
	});
});
