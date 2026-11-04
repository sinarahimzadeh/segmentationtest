import { ChatMistralAI, type ChatMistralAIInput } from "@langchain/mistralai";

type MistralEffort = "none" | "low" | "medium" | "high";

/**
 * LangChain's Mistral adapter 1.2.0 sends images and tool schemas, but does
 * not expose Mistral's reasoning_effort option. Add that one SDK parameter to
 * its normal LangChain request so the selected benchmark level is real.
 */
export class MistralChatModel extends ChatMistralAI {
  private readonly selectedEffort?: MistralEffort;

  constructor(fields: ChatMistralAIInput & { reasoningEffort?: MistralEffort }) {
    super(fields);
    this.selectedEffort = fields.reasoningEffort;
  }

  override invocationParams(options?: this["ParsedCallOptions"]) {
    // The SDK serializes this camelCase field as reasoning_effort. Its current
    // type list lags the documented Large 4 low/medium/high values.
    return {
      ...super.invocationParams(options),
      ...(this.selectedEffort
        ? { reasoningEffort: this.selectedEffort }
        : {}),
    } as ReturnType<ChatMistralAI["invocationParams"]>;
  }
}
