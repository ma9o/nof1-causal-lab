import { parse } from "npyjs";
import type { NumericalArray } from "./generated/models";

export interface DecodedNumericalArray {
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly values: ArrayLike<number | bigint | boolean>;
}

const decoded = new WeakMap<Uint8Array, DecodedNumericalArray>();
const numericalDtypes = new Set([
  "b1",
  "i1",
  "u1",
  "i2",
  "u2",
  "i4",
  "u4",
  "i8",
  "u8",
  "f2",
  "f4",
  "f8",
]);

/** Decode each owned binary buffer once; coordinate selections share its storage. */
export function readNumericalArray(value: NumericalArray): DecodedNumericalArray {
  const existing = decoded.get(value.npy);
  if (existing) return existing;
  // MessagePack's bin offset need not align with a numeric element. The copy owns
  // an aligned ArrayBuffer for NumPy's typed views, including 64-bit integers.
  const array = parse(Uint8Array.from(value.npy).buffer);
  if (!numericalDtypes.has(array.dtype))
    throw new Error(`Unsupported numerical dtype: ${array.dtype}`);
  const result: DecodedNumericalArray = {
    dtype: array.dtype,
    shape: array.shape,
    // npyjs declares ArrayBufferView, but these supported dtypes return indexed
    // typed arrays (or boolean[] for b1), never DataView.
    values: array.data as unknown as ArrayLike<number | bigint | boolean>,
  };
  decoded.set(value.npy, result);
  return result;
}
