# Reduce 性能优化实践

上一篇文章中，我们手写了一个简单的 Reduce 算法，但是性能并不是很好，这一章中我们将会逐步优化这个算法。

+ [交叉寻址](./01_interleaved_addressing/README.md)
+ [解决 Bank Conflict](./02_bank_conflict/README.md)
+ [解决 Idle 线程](./03_idle_thread/README.md)
+ [展开最后一个 warp](./04_unroll_last_warp/README.md)