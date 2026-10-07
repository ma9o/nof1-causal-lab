import { decode, ExtData } from "@msgpack/msgpack";

/** Restore the shared binary values encoded by the Python result codec. */
export function decodeModelMessage(payload: Uint8Array): unknown {
  const decoded = decode(payload);
  const buffers: Uint8Array[] = [];
  function collect(value: unknown): void {
    if (value instanceof Uint8Array) {
      buffers.push(value);
    } else if (Array.isArray(value)) {
      for (const item of value) collect(item);
    } else if (value !== null && typeof value === "object" && !(value instanceof ExtData)) {
      for (const item of Object.values(value)) collect(item);
    }
  }
  collect(decoded);
  // The decoder returns binary views into the payload. Their offsets preserve
  // wire order even when JavaScript enumerates integer object keys differently.
  buffers.sort((left, right) => left.byteOffset - right.byteOffset);

  function restore(value: unknown): unknown {
    if (value instanceof Uint8Array) return value;
    if (value instanceof ExtData) {
      if (value.type !== 42 || !(value.data instanceof Uint8Array) || value.data.byteLength !== 8)
        throw new Error("Unknown numerical MessagePack extension");
      const view = new DataView(value.data.buffer, value.data.byteOffset, value.data.byteLength);
      const index = view.getUint32(0) * 2 ** 32 + view.getUint32(4);
      const buffer = buffers[index];
      if (
        !Number.isSafeInteger(index) ||
        buffer === undefined ||
        buffer.byteOffset >= value.data.byteOffset
      )
        throw new Error("Numerical buffer reference must select an earlier value");
      return buffer;
    }
    if (Array.isArray(value)) return value.map(restore);
    if (value !== null && typeof value === "object")
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, restore(item)]));
    return value;
  }
  return restore(decoded);
}
