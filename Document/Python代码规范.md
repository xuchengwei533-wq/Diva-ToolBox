# PyroEngine Python 代码规范

本篇文章旨在定义 PyroEngine 游戏引擎项目所遵循的 Python 代码标准与开发规范，旨在提高代码的一致性、可读性和可维护性。

## 适用范围
*   工具链脚本（构建/打包/资源处理/部署）
*   数据处理与生成（dataset、标注、评测、统计）
*   训练/推理/评估脚本（若引擎相关 AI 模块使用 Python）
*   示例代码与 CLI 工具

## Import 规范

### 导入顺序（强制）
按以下顺序分组，并在组间空一行：
1.  **标准库**
2.  **第三方库**
3.  **项目内模块**（PyroEngine / repo 内）

**示例：**
```python
from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import requests

from pyroengine.tools.asset import pack_assets
from pyroengine.utils.log import get_logger
```

### 禁止行为
*   ❌ **禁止** `from xxx import *`
*   ❌ **避免** 循环 import；必要时改为局部 import 并说明原因。
*   ✅ 项目内 import 使用 **绝对导入** 为主（从包根开始）。

## 命名规范

### 文件与模块命名
*   **Python 文件名**：`lower_snake_case.py` (下划线连接)
*   **包名**：`lower_snake_case/`
*   **测试文件**：`test_<function>.py`

**示例：**
```text
asset_pipeline.py
model_exporter.py
tests/test_asset_pipeline.py
```

### 类型命名
*   **类名**：`CapWords` (单词首字母大写)

**示例：**
```python
class AssetDatabase:
    ...

class ConfigError(Exception):
    ...
```

### 函数与变量命名
*   **原则**：函数名应该小写，提高可读性可以用下划线分隔。
*   ⚠️ **禁止**：永远不要使用字母 `l`（小写的L），`O`（大写的O），或者 `I`（大写的I）作为单字符变量名。

*   **函数/方法**：`def check_func(): ...`
*   **局部变量**：`lower_snake_case`
*   **私有成员**：前缀 `_`（单下划线）
*   **强私有**：`__name`（双下划线，谨慎使用，仅为了避免命名冲突）

**示例：**
```python
def build_index(asset_root: str) -> dict[str, str]:
    file_count = 0
    ...
```

### 常量命名
*   **原则**：常量默认全部为大写，单词间用下划线分隔，且必须写好注释。

**示例：**
```python
# 默认超时时间（秒）
DEFAULT_TIMEOUT_SEC = 10
MAX_WORKERS = 8
```

## 注释规范
*   **行内注释**：解释 "**为什么**" (Why)，而不是重复代码的 "**是什么**" (What)。
*   **块注释**：复杂逻辑允许分块注释，但要保持简短。

## 代码结构与工程组织

### 推荐目录结构
```text
pyroengine_py/
  pyroengine/
    __init__.py
    tools/
    pipeline/
    utils/
  scripts/
  tests/
  pyproject.toml
```

### 单文件职责
*   一个模块聚焦一个主题（如：`asset pipeline` / `config` / `io` / `logging`）。
*   超过 **300~500 行** 时优先拆分（以可读性为准）。

### 入口（CLI）
*   可执行脚本统一放 `scripts/` 目录。
*   入口函数命名 `main()`，并使用标准入口判断：

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

## 日志（Logging）
*   ❌ **禁止** 使用 `print()` 作为核心逻辑输出（脚本一次性调试除外）。
*   ✅ **使用** `logging` 模块，并统一 logger 名称：
    ```python
    logger = logging.getLogger(__name__)
    ```

### 日志等级规范
*   `debug`：细节信息
*   `info`：关键流程阶段
*   `warning`：可恢复问题
*   `error`：失败但可继续运行
*   `exception`：捕获异常时使用，自动带堆栈信息
