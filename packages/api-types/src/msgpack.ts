import { decode, ExtData } from "@msgpack/msgpack";

/** Restore the shared binary values encoded by the Python result codec. */
export function decodeModelMessage(payload: Uint8Array): unknown {
  const buffers: Uint8Array[] = [];
  function restore(value: unknown): unknown {
    if (value instanceof Uint8Array) {
      buffers.push(value);
      return value;
    }
    if (value instanceof ExtData) {
      if (value.type !== 42 || value.data.byteLength !== 8)
        throw new Error("Unknown numerical MessagePack extension");
      const view = new DataView(value.data.buffer, value.data.byteOffset, value.data.byteLength);
      const index = view.getUint32(0) * 2 ** 32 + view.getUint32(4);
      const buffer = buffers[index];
      if (!Number.isSafeInteger(index) || buffer === undefined)
        throw new Error("Numerical buffer reference must select an earlier value");
      return buffer;
    }
    if (Array.isArray(value)) return value.map(restore);
    if (value !== null && typeof value === "object")
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, restore(item)]));
    return value;
  }
  return restore(decode(payload));
}
