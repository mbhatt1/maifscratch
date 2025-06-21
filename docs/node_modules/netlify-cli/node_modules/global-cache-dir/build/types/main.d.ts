/**
 * Get the global cache directory.
 * Ensures the directory exists.
 * The cache directory location is OS-specific.
 *
 * @example
 * ```js
 * const cacheDir = await globalCacheDir('myapp')
 * // Depending on the OS:
 * //  => '/Users/ehmicky/Library/Caches/myapp'
 * //  => '/home/ehmicky/.cache/myapp'
 * //  => 'C:\Users\ehmicky\AppData\Local\myapp\Cache'
 * ```
 */
export default function globalCacheDir(name: string): Promise<string>
