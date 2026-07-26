import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	getModel,
} from "@earendil-works/pi-ai/compat";
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
		baseDelayMs?: number;
	}) {
		const failCount = options?.failCount ?? 1;
		const maxRetries = options?.maxRetries ?? 0;
		const retryEnabled = options?.retryEnabled ?? false;
		const failStopReason = options?.failStopReason ?? "error";
		const failErrorMessage = options?.failErrorMessage ?? "Connection error.";
		const streamDelayMs = options?.streamDelayMs ?? 0;
		const baseDelayMs = options?.baseDelayMs ?? 1;
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
			retry: { enabled: retryEnabled, maxRetries, baseDelayMs },
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

	it("keeps the failed tail when retry backoff is aborted (Esc), so resume() still works", async () => {
		// Regression: _prepareRetry pops the failed assistant message BEFORE the
		// backoff sleep. If Esc aborts the sleep ("Retry cancelled"), the pop was
		// previously never undone — the tail became the user message and resume()
		// reported "no_failed_turn" even though the turn never completed.
		const created = await createSession({
			failCount: 1,
			retryEnabled: true,
			maxRetries: 3,
			baseDelayMs: 60_000, // long backoff so we can abort mid-sleep
		});

		// Wait deterministically for the backoff sleep to start.
		const backoffStarted = new Promise<void>((resolve) => {
			const unsub = created.session.subscribe((event) => {
				if (event.type === "auto_retry_start") {
					unsub();
					resolve();
				}
			});
		});
		const promptPromise = created.session.prompt("Test");
		await backoffStarted;
		expect(created.session.isRetrying).toBe(true);

		// Esc during backoff → "Retry cancelled".
		created.session.abortRetry();
		await promptPromise;
		expect(created.session.isIdle).toBe(true);

		// The failed assistant message must still be the tail of agent state.
		const messages = created.session.agent.state.messages;
		const tail = messages[messages.length - 1] as AssistantMessage;
		expect(tail.role).toBe("assistant");
		expect(tail.stopReason).toBe("error");

		// /continue must be able to pick it up.
		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });
		expect(created.getCallCount()).toBe(2);
	});

	it("resumes from a user-message tail (process killed mid-stream)", async () => {
		// Crash window: the turn's assistant message was never persisted, so the
		// reloaded session ends with the user message. resume() must re-run it.
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Setup"); // healthy completed turn
		expect(created.getCallCount()).toBe(1);

		// Simulate post-crash reload: tail is an unanswered user message.
		created.session.agent.state.messages = [
			...created.session.agent.state.messages,
			{ role: "user", content: [{ type: "text", text: "Unanswered" }], timestamp: Date.now() },
		];

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });
		expect(created.getCallCount()).toBe(2);

		const tail = created.session.agent.state.messages.at(-1) as AssistantMessage;
		expect(tail.role).toBe("assistant");
		expect(tail.stopReason).toBe("stop");
	});

	it("resumes from a toolUse tail by synthesizing missing toolResults (killed during tools)", async () => {
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Setup");

		// Simulate post-crash reload: assistant asked for two parallel tool calls,
		// process died before any tool result was recorded.
		const toolUseMsg = createAssistantMessage("", {
			stopReason: "toolUse",
			content: [
				{ type: "toolCall", id: "call_1", name: "bash", arguments: { command: "ls" } },
				{ type: "toolCall", id: "call_2", name: "read", arguments: { path: "x" } },
			],
		});
		created.session.agent.state.messages = [...created.session.agent.state.messages, toolUseMsg];

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });

		// Two synthesized error toolResults, inserted directly after the toolUse
		// assistant message so providers see valid tool_use/tool_result ordering.
		const state = created.session.agent.state.messages;
		const toolUseIdx = state.findIndex((m) => m === toolUseMsg);
		const r1 = state[toolUseIdx + 1];
		const r2 = state[toolUseIdx + 2];
		expect(r1.role).toBe("toolResult");
		expect(r2.role).toBe("toolResult");
		expect((r1 as { toolCallId: string }).toolCallId).toBe("call_1");
		expect((r2 as { toolCallId: string }).toolCallId).toBe("call_2");
		expect((r1 as { isError: boolean }).isError).toBe(true);

		// Synthesized results are persisted — a future reload must not replay
		// dangling tool calls.
		const entries = created.session.sessionManager.getEntries();
		const persisted = entries.filter(
			(e) => e.type === "message" && (e.message as { role: string }).role === "toolResult",
		);
		expect(persisted.length).toBe(2);

		// The continuation itself ran to completion.
		expect(created.getCallCount()).toBe(2);
		const tail = state.at(-1) as AssistantMessage;
		expect(tail.stopReason).toBe("stop");
	});

	it("resumes from a toolResult tail (killed between tool result and next LLM call)", async () => {
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Setup");

		const toolUseMsg = createAssistantMessage("", {
			stopReason: "toolUse",
			content: [{ type: "toolCall", id: "call_1", name: "bash", arguments: {} }],
		});
		created.session.agent.state.messages = [
			...created.session.agent.state.messages,
			toolUseMsg,
			{
				role: "toolResult",
				toolCallId: "call_1",
				toolName: "bash",
				content: [{ type: "text", text: "done" }],
				isError: false,
				timestamp: Date.now(),
			},
		];

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });
		expect(created.getCallCount()).toBe(2);

		// No dangling calls here — nothing should have been synthesized.
		const synthesized = created.session.agent.state.messages.filter(
			(m) => m.role === "toolResult" && (m as { isError: boolean }).isError,
		);
		expect(synthesized.length).toBe(0);
	});

	it("resumes a failed turn even when bash messages trail the failed tail", async () => {
		// _flushPendingBashMessages runs in the finally of a failed turn, so a
		// bashExecution message can sit AFTER the failed assistant message.
		const created = await createSession({ failCount: 1, retryEnabled: false });
		await created.session.prompt("Test");
		expect(created.getCallCount()).toBe(1);

		created.session.agent.state.messages = [
			...created.session.agent.state.messages,
			{
				role: "bashExecution",
				command: "ls",
				output: "file.txt",
				exitCode: 0,
				cancelled: false,
				truncated: false,
				timestamp: Date.now(),
			},
		];

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });
		expect(created.getCallCount()).toBe(2);

		// Error popped from the middle; bash message and new reply both present.
		const state = created.session.agent.state.messages;
		expect(state.some((m) => m.role === "bashExecution")).toBe(true);
		expect(state.some((m) => m.role === "assistant" && (m as AssistantMessage).stopReason === "error")).toBe(
			false,
		);
		expect((state.at(-1) as AssistantMessage).stopReason).toBe("stop");
	});

	it("still refuses when a completed turn is followed only by bash output", async () => {
		// Healthy idle session: user ran `!ls` after the turn finished. The tail
		// is bashExecution but the effective LLM tail is stopReason "stop".
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Test");
		created.session.agent.state.messages = [
			...created.session.agent.state.messages,
			{
				role: "bashExecution",
				command: "ls",
				output: "file.txt",
				exitCode: 0,
				cancelled: false,
				truncated: false,
				timestamp: Date.now(),
			},
		];

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: false, reason: "no_failed_turn" });
	});

	it("resumes from an empty thinking-only stop tail (provider truncation)", async () => {
		// Some providers cut the completion server-side and return a "stop"
		// response with thinking content but no text and 0 output tokens. The
		// user sees nothing — the turn is degenerate and must be re-runnable.
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Setup");

		const emptyStop = createAssistantMessage("", {
			stopReason: "stop",
			content: [{ type: "thinking", thinking: "Let me consider...", thinkingSignature: "sig" }],
		});
		created.session.agent.state.messages = [
			...created.session.agent.state.messages,
			{ role: "user", content: [{ type: "text", text: "Unanswered" }], timestamp: Date.now() },
			emptyStop,
		];

		const result = await created.session.resume();
		expect(result).toEqual({ resumed: true });
		expect(created.getCallCount()).toBe(2);

		// Empty completion popped; new reply is the tail.
		const state = created.session.agent.state.messages;
		expect(state.some((m) => m === emptyStop)).toBe(false);
		expect((state.at(-1) as AssistantMessage).stopReason).toBe("stop");
		expect(((state.at(-1) as AssistantMessage).content[0] as { text: string }).text).toBe("Success after resume");
	});

	it("still refuses a stop tail that has real text", async () => {
		const created = await createSession({ failCount: 0, retryEnabled: false });
		await created.session.prompt("Test"); // completes with visible text
		const result = await created.session.resume();
		expect(result).toEqual({ resumed: false, reason: "no_failed_turn" });
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
