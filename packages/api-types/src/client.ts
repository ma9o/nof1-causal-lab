import createClient, { type ClientOptions } from "openapi-fetch";
import type { paths } from "./generated/model-api";

/** The same typed read client works directly against Python or through the Next.js proxy. */
export function createModelClient(options?: ClientOptions) {
  return createClient<paths>(options);
}
