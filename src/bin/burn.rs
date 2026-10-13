// AVX-512 / AVX2 の FMA を全スレッドで回し続けて、CPU の電力・発熱を最大化する
#![allow(unsafe_op_in_unsafe_fn)]
use std::arch::x86_64::{__m256d, __m512d};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const INNER: u64 = 1 << 16;

// 8 レーン × 16 本の独立した FMA チェーン
#[target_feature(enable = "avx512f")]
unsafe fn burn512(stop: &AtomicBool) -> (u64, f64) {
    let a: __m512d = std::mem::transmute([0.999_999f64; 8]);
    let b: __m512d = std::mem::transmute([1e-6f64; 8]);
    let mut acc: [__m512d; 16] = std::mem::transmute([[1.0f64; 8]; 16]);
    let mut iters = 0u64;
    while !stop.load(Ordering::Relaxed) {
        for _ in 0..INNER {
            // コンパイラに消されないよう FMA を直接発行する
            unsafe {
                std::arch::asm!(
                "vfmadd213pd {a0}, {m}, {b}",
                "vfmadd213pd {a1}, {m}, {b}",
                "vfmadd213pd {a2}, {m}, {b}",
                "vfmadd213pd {a3}, {m}, {b}",
                "vfmadd213pd {a4}, {m}, {b}",
                "vfmadd213pd {a5}, {m}, {b}",
                "vfmadd213pd {a6}, {m}, {b}",
                "vfmadd213pd {a7}, {m}, {b}",
                "vfmadd213pd {a8}, {m}, {b}",
                "vfmadd213pd {a9}, {m}, {b}",
                "vfmadd213pd {a10}, {m}, {b}",
                "vfmadd213pd {a11}, {m}, {b}",
                "vfmadd213pd {a12}, {m}, {b}",
                "vfmadd213pd {a13}, {m}, {b}",
                "vfmadd213pd {a14}, {m}, {b}",
                "vfmadd213pd {a15}, {m}, {b}",
                a0 = inout(zmm_reg) acc[0],
                a1 = inout(zmm_reg) acc[1],
                a2 = inout(zmm_reg) acc[2],
                a3 = inout(zmm_reg) acc[3],
                a4 = inout(zmm_reg) acc[4],
                a5 = inout(zmm_reg) acc[5],
                a6 = inout(zmm_reg) acc[6],
                a7 = inout(zmm_reg) acc[7],
                a8 = inout(zmm_reg) acc[8],
                a9 = inout(zmm_reg) acc[9],
                a10 = inout(zmm_reg) acc[10],
                a11 = inout(zmm_reg) acc[11],
                a12 = inout(zmm_reg) acc[12],
                a13 = inout(zmm_reg) acc[13],
                a14 = inout(zmm_reg) acc[14],
                a15 = inout(zmm_reg) acc[15],
                    m = in(zmm_reg) a,
                    b = in(zmm_reg) b,
                    options(nomem, nostack),
                );
            }
        }
        iters += INNER;
    }
    let s = std::mem::transmute::<_, [[f64; 8]; 16]>(acc).iter().flatten().sum();
    (iters * 16 * 8 * 2, s)
}

// 4 レーン × 12 本の独立した FMA チェーン
#[target_feature(enable = "avx2,fma")]
unsafe fn burn256(stop: &AtomicBool) -> (u64, f64) {
    let a: __m256d = std::mem::transmute([0.999_999f64; 4]);
    let b: __m256d = std::mem::transmute([1e-6f64; 4]);
    let mut acc: [__m256d; 12] = std::mem::transmute([[1.0f64; 4]; 12]);
    let mut iters = 0u64;
    while !stop.load(Ordering::Relaxed) {
        for _ in 0..INNER {
            // コンパイラに消されないよう FMA を直接発行する
            unsafe {
                std::arch::asm!(
                "vfmadd213pd {a0}, {m}, {b}",
                "vfmadd213pd {a1}, {m}, {b}",
                "vfmadd213pd {a2}, {m}, {b}",
                "vfmadd213pd {a3}, {m}, {b}",
                "vfmadd213pd {a4}, {m}, {b}",
                "vfmadd213pd {a5}, {m}, {b}",
                "vfmadd213pd {a6}, {m}, {b}",
                "vfmadd213pd {a7}, {m}, {b}",
                "vfmadd213pd {a8}, {m}, {b}",
                "vfmadd213pd {a9}, {m}, {b}",
                "vfmadd213pd {a10}, {m}, {b}",
                "vfmadd213pd {a11}, {m}, {b}",
                a0 = inout(ymm_reg) acc[0],
                a1 = inout(ymm_reg) acc[1],
                a2 = inout(ymm_reg) acc[2],
                a3 = inout(ymm_reg) acc[3],
                a4 = inout(ymm_reg) acc[4],
                a5 = inout(ymm_reg) acc[5],
                a6 = inout(ymm_reg) acc[6],
                a7 = inout(ymm_reg) acc[7],
                a8 = inout(ymm_reg) acc[8],
                a9 = inout(ymm_reg) acc[9],
                a10 = inout(ymm_reg) acc[10],
                a11 = inout(ymm_reg) acc[11],
                    m = in(ymm_reg) a,
                    b = in(ymm_reg) b,
                    options(nomem, nostack),
                );
            }
        }
        iters += INNER;
    }
    let s = std::mem::transmute::<_, [[f64; 4]; 12]>(acc).iter().flatten().sum();
    (iters * 12 * 4 * 2, s)
}

// 電力ウイルス型: ランダムな仮数で FMA し続け、加算側は L1 キャッシュから読む。
// ビットの切り替わりが多いほど消費電力が上がる。
#[target_feature(enable = "avx512f")]
unsafe fn burn_virus(stop: &AtomicBool, seed: u64) -> (u64, f64) {
    let mut x = seed | 1;
    let mut rnd = || {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        (x >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
    };
    // 乗数は |a| が 0.5〜0.95 で符号もランダム → 値が発散も収束もせず散らばり続ける
    let mut mul = [[0f64; 8]; 4];
    for v in mul.iter_mut().flatten() {
        *v = (0.5 + 0.45 * rnd()) * if rnd() < 0.5 { -1.0 } else { 1.0 };
    }
    #[repr(align(64))]
    struct Buf([[f64; 8]; 64]); // 4 KiB: L1 に収まる
    let mut buf = Buf([[0f64; 8]; 64]);
    for v in buf.0.iter_mut().flatten() { *v = rnd() * 2.0 - 1.0; }
    let mut acc = [[0f64; 8]; 16];
    for v in acc.iter_mut().flatten() { *v = rnd(); }
    let mul: [__m512d; 4] = std::mem::transmute(mul);
    let mut acc: [__m512d; 16] = std::mem::transmute(acc);
    let p = buf.0.as_ptr();
    let mut iters = 0u64;
    while !stop.load(Ordering::Relaxed) {
        for _ in 0..INNER / 4 {
            std::arch::asm!(
                    "vfmadd213pd {a0}, {m0}, [{p} + 0]",
                    "vfmadd213pd {a1}, {m1}, [{p} + 64]",
                    "vfmadd213pd {a2}, {m2}, [{p} + 128]",
                    "vfmadd213pd {a3}, {m3}, [{p} + 192]",
                    "vfmadd213pd {a4}, {m0}, [{p} + 256]",
                    "vfmadd213pd {a5}, {m1}, [{p} + 320]",
                    "vfmadd213pd {a6}, {m2}, [{p} + 384]",
                    "vfmadd213pd {a7}, {m3}, [{p} + 448]",
                    "vfmadd213pd {a8}, {m0}, [{p} + 512]",
                    "vfmadd213pd {a9}, {m1}, [{p} + 576]",
                    "vfmadd213pd {a10}, {m2}, [{p} + 640]",
                    "vfmadd213pd {a11}, {m3}, [{p} + 704]",
                    "vfmadd213pd {a12}, {m0}, [{p} + 768]",
                    "vfmadd213pd {a13}, {m1}, [{p} + 832]",
                    "vfmadd213pd {a14}, {m2}, [{p} + 896]",
                    "vfmadd213pd {a15}, {m3}, [{p} + 960]",
                    "vfmadd213pd {a0}, {m1}, [{p} + 1024]",
                    "vfmadd213pd {a1}, {m2}, [{p} + 1088]",
                    "vfmadd213pd {a2}, {m3}, [{p} + 1152]",
                    "vfmadd213pd {a3}, {m0}, [{p} + 1216]",
                    "vfmadd213pd {a4}, {m1}, [{p} + 1280]",
                    "vfmadd213pd {a5}, {m2}, [{p} + 1344]",
                    "vfmadd213pd {a6}, {m3}, [{p} + 1408]",
                    "vfmadd213pd {a7}, {m0}, [{p} + 1472]",
                    "vfmadd213pd {a8}, {m1}, [{p} + 1536]",
                    "vfmadd213pd {a9}, {m2}, [{p} + 1600]",
                    "vfmadd213pd {a10}, {m3}, [{p} + 1664]",
                    "vfmadd213pd {a11}, {m0}, [{p} + 1728]",
                    "vfmadd213pd {a12}, {m1}, [{p} + 1792]",
                    "vfmadd213pd {a13}, {m2}, [{p} + 1856]",
                    "vfmadd213pd {a14}, {m3}, [{p} + 1920]",
                    "vfmadd213pd {a15}, {m0}, [{p} + 1984]",
                    "vfmadd213pd {a0}, {m2}, [{p} + 2048]",
                    "vfmadd213pd {a1}, {m3}, [{p} + 2112]",
                    "vfmadd213pd {a2}, {m0}, [{p} + 2176]",
                    "vfmadd213pd {a3}, {m1}, [{p} + 2240]",
                    "vfmadd213pd {a4}, {m2}, [{p} + 2304]",
                    "vfmadd213pd {a5}, {m3}, [{p} + 2368]",
                    "vfmadd213pd {a6}, {m0}, [{p} + 2432]",
                    "vfmadd213pd {a7}, {m1}, [{p} + 2496]",
                    "vfmadd213pd {a8}, {m2}, [{p} + 2560]",
                    "vfmadd213pd {a9}, {m3}, [{p} + 2624]",
                    "vfmadd213pd {a10}, {m0}, [{p} + 2688]",
                    "vfmadd213pd {a11}, {m1}, [{p} + 2752]",
                    "vfmadd213pd {a12}, {m2}, [{p} + 2816]",
                    "vfmadd213pd {a13}, {m3}, [{p} + 2880]",
                    "vfmadd213pd {a14}, {m0}, [{p} + 2944]",
                    "vfmadd213pd {a15}, {m1}, [{p} + 3008]",
                    "vfmadd213pd {a0}, {m3}, [{p} + 3072]",
                    "vfmadd213pd {a1}, {m0}, [{p} + 3136]",
                    "vfmadd213pd {a2}, {m1}, [{p} + 3200]",
                    "vfmadd213pd {a3}, {m2}, [{p} + 3264]",
                    "vfmadd213pd {a4}, {m3}, [{p} + 3328]",
                    "vfmadd213pd {a5}, {m0}, [{p} + 3392]",
                    "vfmadd213pd {a6}, {m1}, [{p} + 3456]",
                    "vfmadd213pd {a7}, {m2}, [{p} + 3520]",
                    "vfmadd213pd {a8}, {m3}, [{p} + 3584]",
                    "vfmadd213pd {a9}, {m0}, [{p} + 3648]",
                    "vfmadd213pd {a10}, {m1}, [{p} + 3712]",
                    "vfmadd213pd {a11}, {m2}, [{p} + 3776]",
                    "vfmadd213pd {a12}, {m3}, [{p} + 3840]",
                    "vfmadd213pd {a13}, {m0}, [{p} + 3904]",
                    "vfmadd213pd {a14}, {m1}, [{p} + 3968]",
                    "vfmadd213pd {a15}, {m2}, [{p} + 4032]",
                    a0 = inout(zmm_reg) acc[0],
                    a1 = inout(zmm_reg) acc[1],
                    a2 = inout(zmm_reg) acc[2],
                    a3 = inout(zmm_reg) acc[3],
                    a4 = inout(zmm_reg) acc[4],
                    a5 = inout(zmm_reg) acc[5],
                    a6 = inout(zmm_reg) acc[6],
                    a7 = inout(zmm_reg) acc[7],
                    a8 = inout(zmm_reg) acc[8],
                    a9 = inout(zmm_reg) acc[9],
                    a10 = inout(zmm_reg) acc[10],
                    a11 = inout(zmm_reg) acc[11],
                    a12 = inout(zmm_reg) acc[12],
                    a13 = inout(zmm_reg) acc[13],
                    a14 = inout(zmm_reg) acc[14],
                    a15 = inout(zmm_reg) acc[15],
                    m0 = in(zmm_reg) mul[0],
                    m1 = in(zmm_reg) mul[1],
                    m2 = in(zmm_reg) mul[2],
                    m3 = in(zmm_reg) mul[3],
                    p = in(reg) p,
                    options(readonly, nostack),
            );
        }
        iters += INNER / 4;
    }
    let s = std::mem::transmute::<_, [[f64; 8]; 16]>(acc).iter().flatten().map(|v| v.abs()).sum();
    (iters * 64 * 8 * 2, s)
}

// 64 MiB のバッファを読み書きし続けて、メモリ帯域とメモリコントローラを使い切る
fn mem_worker(stop: &AtomicBool) -> u64 {
    let mut buf = vec![1u64; 8 << 20];
    let mut bytes = 0u64;
    let mut k = 3u64;
    while !stop.load(Ordering::Relaxed) {
        for v in buf.iter_mut() {
            *v = v.wrapping_mul(k).wrapping_add(1);
        }
        k = black_box(k);
        bytes += (buf.len() * 8 * 2) as u64; // 読み + 書き
    }
    black_box(&buf);
    bytes
}

fn main() {
    // 使い方: burn [秒数=60] [スレッド数=0(全部)] [avx2|mix|virus]
    //   virus: ランダム値 + L1 読み込みの AVX-512 FMA (最大電力狙い)
    //   mix: 半分のスレッドで AVX-512 FMA、残り半分でメモリ読み書き
    let args: Vec<String> = std::env::args().collect();
    let secs: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).filter(|&n| n > 0)
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let mode = args.get(3).map(|s| s.as_str()).unwrap_or("");
    let mix = mode == "mix";
    let virus = mode == "virus" && is_x86_feature_detected!("avx512f");
    let use512 = is_x86_feature_detected!("avx512f") && mode != "avx2";
    if !use512 && !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
        eprintln!("AVX2/FMA 非対応の CPU です");
        return;
    }
    let n_mem = if mix { n / 2 } else { 0 };
    println!("{} FMA {} スレッド + メモリ {n_mem} スレッドで {secs} 秒間回します (Ctrl+C で中断)",
             if virus { "AVX-512 virus" } else if use512 { "AVX-512" } else { "AVX2" }, n - n_mem);

    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let handles: Vec<_> = (0..n).map(|i| {
        let stop = stop.clone();
        thread::spawn(move || unsafe {
            if i >= n - n_mem { (0, 0.0, mem_worker(&stop)) }
            else {
                let (f, s) = if virus { burn_virus(&stop, 0x9E37_79B9_7F4A_7C15 ^ i as u64) } else if use512 { burn512(&stop) } else { burn256(&stop) };
                (f, s, 0)
            }
        })
    }).collect();

    for t in 1..=secs {
        thread::sleep(Duration::from_secs(1));
        if t % 10 == 0 { println!("  {t} 秒経過"); }
    }
    stop.store(true, Ordering::Relaxed);
    let (mut flops, mut chk, mut bytes) = (0u64, 0.0, 0u64);
    for h in handles {
        let (f, s, b) = h.join().unwrap();
        flops += f;
        chk += s;
        bytes += b;
    }
    let el = start.elapsed().as_secs_f64();
    println!("合計 {:.2} TFLOP / {el:.1} 秒 = {:.1} GFLOPS (検算値 {:.3})",
             flops as f64 / 1e12, flops as f64 / el / 1e9, black_box(chk));
    if n_mem > 0 {
        println!("メモリ {:.1} GB/s", bytes as f64 / el / 1e9);
    }
}
