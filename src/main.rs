use std::thread;
use std::time::Instant;

// 素朴な竹内関数。呼び出し回数を数える（遅延評価もメモ化もしない）
fn tarai(x: i32, y: i32, z: i32, calls: &mut u64) -> i32 {
    *calls += 1;
    if x <= y {
        return y;
    }
    let a = tarai(x - 1, y, z, calls);
    let b = tarai(y - 1, z, x, calls);
    let c = tarai(z - 1, x, y, calls);
    tarai(a, b, c, calls)
}

fn main() {
    // 使い方: tarai <x> [スレッド数]   → 各スレッドが tarai(x, x/2, 0) を計算
    let args: Vec<String> = std::env::args().collect();
    let x: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(16);
    let n: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| thread::available_parallelism().map(|n| n.get()).unwrap_or(1));

    println!("tarai({x}, {}, 0) を {n} スレッドで同時に計算します…", x / 2);
    let start = Instant::now();
    let handles: Vec<_> = (0..n)
        .map(|i| {
            thread::spawn(move || {
                let mut calls = 0u64;
                let t = Instant::now();
                let r = tarai(x, x / 2, 0, &mut calls);
                println!("  thread {i:>2}: 結果 {r}  呼び出し {calls} 回  {:.2?}", t.elapsed());
                calls
            })
        })
        .collect();
    let total: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let secs = start.elapsed().as_secs_f64();
    println!("合計 {total} 回の関数呼び出し / {secs:.2} 秒 ({:.0} 百万回/秒)", total as f64 / secs / 1e6);
}
