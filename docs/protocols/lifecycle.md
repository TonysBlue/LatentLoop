# 服务生命周期

每个 Model Service/Harness session 都遵循：

```text
identity -> open_session -> reset -> ordered units -> evaluate -> close_session
```

任何 unit、screen revision 或 session 顺序错误都必须 fail closed。
