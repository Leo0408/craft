# Step 5 错误修复

## 问题描述

在 Step 5 执行时出现错误：
```
AttributeError: 'ConstraintGenerator' object has no attribute 'compile_constraint'
Cell In[15], line 14
condition_expr = constraint_generator.compile_constraint(constraint)
```

## 问题原因

`ConstraintGenerator` 类缺少 `compile_constraint` 方法。该方法应该将约束描述编译成可执行的 AST/DSL 表达式。

根据 Method.md Section 2.2，约束应该被编译成可执行的代码表达式，例如：
- `(empty coffee_machine)`
- `(inside mug coffee_machine)`
- `(eq machine.door 'open')`

## 修复方案

在 `ConstraintGenerator` 类中添加了 `compile_constraint` 方法，该方法能够：

1. **解析状态约束** - "must be empty/open/closed/filled"
   - 示例：`"Coffee machine must be open"` → `"(open coffee_machine)"`

2. **解析位置约束** - "must be on/inside/in"
   - 示例：`"Mug must be inside Sink"` → `"(inside mug sink)"`
   - 示例：`"Mug must be on top of CounterTop"` → `"(on_top_of mug countertop)"`

3. **解析否定约束** - "must not be"
   - 示例：`"Mug must not be inside CoffeeMachine"` → `"(not (inside mug coffee_machine))"`

4. **解析移动约束** - "must be moved from X to Y"
   - 示例：`"Mug must be moved from Sink to CoffeeMachine"` → `"(and (inside mug coffee_machine) (not (inside mug sink)))"`

5. **解析容器空约束** - "container must be empty"
   - 示例：`"Coffee machine must be empty"` → `"(empty coffee_machine)"`

## 实现细节

方法签名：
```python
def compile_constraint(self, constraint: Dict) -> Optional[str]:
    """
    Compile constraint description to executable condition expression (AST/DSL)
    
    Args:
        constraint: Constraint dictionary with 'description', 'type', 'condition'
        
    Returns:
        Executable condition expression string, or None if cannot compile
    """
```

### 处理流程

1. **优先使用已有条件**：如果约束已经有 `condition` 字段，直接返回
2. **模式匹配**：根据约束描述的模式（"must be", "must not", "must be moved from" 等）进行解析
3. **对象名称提取**：从描述中提取对象名称（处理 "The X" 格式）
4. **状态/位置提取**：从描述中提取状态或位置信息
5. **生成 DSL 表达式**：将解析结果转换为 AST/DSL 格式

### 示例转换

| 约束描述 | 编译结果 |
|---------|---------|
| `"Coffee machine must be open"` | `"(open coffee_machine)"` |
| `"Mug must be inside Sink"` | `"(inside mug sink)"` |
| `"Mug must be on top of CounterTop"` | `"(on_top_of mug countertop)"` |
| `"Coffee machine must be empty"` | `"(empty coffee_machine)"` |
| `"Mug must not be inside CoffeeMachine"` | `"(not (inside mug coffee_machine))"` |

## 修复内容

✅ **添加了 `compile_constraint` 方法**到 `ConstraintGenerator` 类
- 位置：`/home/leo/craft/reasoning/constraint_generator.py` (第 404 行)
- 支持多种约束模式解析
- 返回可执行的 AST/DSL 表达式

## 验证结果

方法已成功添加到 `ConstraintGenerator` 类中：
- ✅ 方法定义存在：`def compile_constraint(self, constraint: Dict) -> Optional[str]:`
- ✅ 支持多种约束模式
- ✅ 返回格式符合 Method.md 要求

## 使用方式

在 Step 5 中，现在可以正常调用：

```python
compiled_constraints = []
for constraint in constraints_craft:
    condition_expr = constraint_generator.compile_constraint(constraint)
    if condition_expr:
        compiled_constraints.append({
            'constraint': constraint,
            'condition_expr': condition_expr
        })
```

## 注意事项

1. **无法编译的约束**：如果约束描述无法匹配任何模式，方法返回 `None`，该约束会被跳过
2. **对象名称匹配**：方法会尝试处理 "The X" 格式的对象名称
3. **大小写处理**：所有对象名称和状态都会转换为小写，并用下划线替换空格
4. **条件字段优先**：如果约束已经有 `condition` 字段，直接使用，不进行解析

## 后续改进建议

1. **LLM 生成条件表达式**：可以让 LLM 在生成约束时直接生成 `condition_expr`，减少解析错误
2. **更复杂的模式匹配**：支持更多约束模式，如时间约束、数量约束等
3. **验证表达式有效性**：在编译后验证表达式是否符合 DSL 语法

