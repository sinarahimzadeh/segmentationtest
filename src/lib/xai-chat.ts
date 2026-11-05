import {
  ChatXAIResponses,
  type XAIResponse,
  type XAIResponsesCreateParamsNonStreaming,
  type XAIResponsesCreateParamsStreaming,
  type XAIResponsesStreamEvent,
} from "@langchain/xai";

/**
 * LangChain's xAI message converter currently replaces image detail with
 * "auto". Keep the app's existing "high" setting for all eight scan views.
 */
export class ScanbodyChatXAI extends ChatXAIResponses {
  protected override _makeRequest(
    request: XAIResponsesCreateParamsNonStreaming,
  ): Promise<XAIResponse>;
  protected override _makeRequest(
    request: XAIResponsesCreateParamsStreaming,
  ): Promise<AsyncIterable<XAIResponsesStreamEvent>>;
  protected override _makeRequest(
    request: XAIResponsesCreateParamsNonStreaming | XAIResponsesCreateParamsStreaming,
  ): Promise<XAIResponse> | Promise<AsyncIterable<XAIResponsesStreamEvent>> {
    const input =
      typeof request.input === "string"
        ? request.input
        : request.input.map((item) =>
            "content" in item && Array.isArray(item.content)
              ? {
                  ...item,
                  content: item.content.map((part) =>
                    part.type === "input_image"
                      ? { ...part, detail: "high" as const }
                      : part,
                  ),
                }
              : item,
          );

    if (request.stream) return super._makeRequest({ ...request, input });
    return super._makeRequest({ ...request, input });
  }
}
