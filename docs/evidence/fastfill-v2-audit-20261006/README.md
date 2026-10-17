# 审核证据的使用范围

本目录随 GitHub 保存可阅读的结论、逐文件阅读/hash清单、CPU/整数机制反例、最终测试与依赖检查摘要。它是完整本地审核证据的便携子集；不是可以独立通过上游 bundle verifier 的完整 bundle。

原始论文 PDF、页面图、完整 stdout、source-read-copy、coverage 原文件和合成 GLB fixtures 保留在 `outputs/fastfill_v2/audit-four-20261006/`，并随服务器审核目录同步。为避免在 Git 中重复上游源代码和论文全文，本目录没有复制它们。部分原上游 artifact-manifest 的范围比本便携子集大，必须在原证据目录运行对应 verifier；本目录自己的 [artifact-manifest.json](artifact-manifest.json) 只列实际携带文件。

参考源代码在固定 Git submodules；论文来源与 SHA 在各 paper-manifest/receipt 中。源码全文阅读不等于原 detector/CUDA 实跑；Minkowski native 反例为整数机制模拟；RoomGenBench GLB 回归为合成 fixture，不声称 learned generation、物理或 Host 完成。
