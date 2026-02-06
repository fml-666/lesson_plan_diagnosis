import streamlit as st
from zhipuai import ZhipuAI  # 智谱GLM-4库
import json

# ---------------------- 1. 配置大模型（智谱GLM-4）----------------------
API_KEY = st.secrets["api_key"]
client = ZhipuAI(api_key=API_KEY)


# ---------------------- 2. 大模型调用函数----------------------
def model_invocation(prompt):
    """用智谱GLM-4分析教案，返回解析后的字典，增强容错处理"""
    try:
        response = client.chat.completions.create(
            model="glm-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000
        )
        content = response.choices[0].message.content

        # 保存原始响应以便调试
        raw_content = content

        # 清洗1: 去除 Markdown 代码块标记 (```json ... ```)
        content = content.strip()
        if content.startswith("```"):
            # 找到第一个换行符，去掉第一行(```json)
            first_newline = content.find('\n')
            if first_newline != -1:
                content = content[first_newline + 1:]
            # 去掉末尾的 ```
            if content.endswith("```"):
                content = content[:-3].strip()

        # 清洗2: 查找第一个 { 和最后一个 } 之间的内容（防止开头有文字说明）
        start_idx = content.find('{')
        end_idx = content.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx + 1]

        # 清洗3: 去除首尾空白
        content = content.strip()

        # 尝试解析
        return json.loads(content)

    except json.JSONDecodeError as e:
        # 如果还是失败，返回包含原始内容的错误信息，方便调试
        return {
            "error": "大模型返回内容不是合法JSON格式",
            "解析错误": str(e),
            "原始内容前200字": raw_content[:200] if 'raw_content' in locals() else "无",
            "清洗后内容前200字": content[:200] if 'content' in locals() else "无"
        }
    except Exception as e:
        return {"error": f"调用大模型失败：{str(e)}"}


# ---------------------- 3. 教案检测核心功能----------------------
def test_lesson_plan(text):
    """分三步调用，降低单次token消耗，提高准确性"""

    # 第一步：环节完整性检测（精简版）
    step1_result = check_completeness(text)

    # 第二步：时间分配检测（基于step1结果）
    step2_result = check_time_allocation(text, step1_result)

    # 第三步：素养匹配检测（独立进行）
    step3_result = check_literacy(text)

    # 合并结果
    return {
        "环节完整性": step1_result,
        "时间分配": step2_result,
        "素养匹配": step3_result
    }


def check_completeness(text):
    """环节完整性检测 - 约1500 tokens"""
    prompt = f"""你是初中信息科技教案诊断专家，检测教学环节完整性。

【5环节识别关键词】
导入：导入/引入/情境导入/问题导入/激趣导入/开场
新授：新授/新课/讲授/讲解/知识讲解/新知探究
练习：练习/实践/活动/探究活动/动手实践/巩固练习
总结：总结/小结/课堂总结/归纳/知识梳理/回顾
作业：作业/作业布置/课后作业/课堂作业/家庭作业/拓展作业/课后练习/课后任务/布置作业/课后延伸/课外作业/五、作业/六、作业

【检测规则】
1. 逐段扫描全文，合并分散的同类活动
2. 作业环节重点检查：教案末尾、编号段落（五、六、七）
3. 字数≥30字且有具体操作=内容有效

【评分】每环节20分，缺失0分，存在但简略（<30字）10分，完整20分

输出JSON：
{{
"score": "0-100",
"详情": "扣分说明",
"缺失环节": ["作业"]或[],
"各环节状态": [
  {{"环节":"导入","得分":20,"是否存在":true,"内容有效":true,"摘要":"..."}},
  {{"环节":"新授","得分":20,"是否存在":true,"内容有效":true,"摘要":"..."}},
  {{"环节":"练习","得分":10,"是否存在":true,"内容有效":false,"摘要":"..."}},
  {{"环节":"总结","得分":20,"是否存在":true,"内容有效":true,"摘要":"..."}},
  {{"环节":"作业","得分":0,"是否存在":false,"内容有效":false,"摘要":""}}
]
}}

教案：{text}
"""
    return model_invocation(prompt)


def check_time_allocation(text, completeness_result):
    """时间分配检测 - 约1200 tokens"""
    # 提取环节存在情况
    existing = [s["环节"] for s in completeness_result.get("各环节状态", []) if s.get("是否存在")]

    prompt = f"""你是教案时间分析专家，检测时间分配合理性。

【已确认存在的环节】{existing}
【总课时】40分钟

【时间推断规则】无明确标注时智能推断：
- 导入：简单提问3min，多媒体5min，复杂情境7min
- 新授：1-2概念12-15min，3-4知识点18-22min，复杂+互动23-25min  
- 练习：简单操作5-8min，小组讨论10-12min，复杂项目15-18min
- 总结：教师总结2min，互动3min，学生归纳4-5min
- 作业：简单题目2min，实践任务3-4min

【合理范围】
导入1-7min，新授10-25min，练习5-18min，总结1-5min，作业1-4min

【输出格式】
"当前时长"标注来源：X分钟[原文标注] 或 X分钟[推断：依据...] 或 0[环节缺失]

输出JSON：
{{
"score": "0-100",
"详情": "含推断逻辑说明",
"建议": [
  {{"环节":"导入","当前时长":"5分钟[原文标注]","建议时长":"1-7分钟","是否合理":true}},
  {{"环节":"新授","当前时长":"22分钟[推断：3个知识点+演示]","建议时长":"10-25分钟","是否合理":true}},
  ...
]
}}

教案：{text}
"""
    return model_invocation(prompt)


def check_literacy(text):
    """素养匹配检测 - 约1800 tokens（保持原逻辑但精简）"""
    prompt = f"""你是核心素养评估专家，用锚点对比法评估四维素养。

【锚点对比法】最终分=优秀匹配×0.5+基础匹配×0.3+(1-缺失匹配)×0.2

【四维锚点】
信息意识：
- 优秀：对比官方/自媒体信息，验证真实性，记录依据
- 基础：辨别网络信息真伪，培养甄别意识
- 缺失：仅学习使用搜索引擎

计算思维：
- 优秀：任务分解为步骤，绘制流程图，描述算法逻辑
- 基础：按步骤完成操作，理解问题解决过程
- 缺失：仅记住操作步骤

数字化学习与创新：
- 优秀：小组协作在线文档，插入多媒体，在线演示
- 基础：利用Word制作小报，整合网络资料
- 缺失：仅阅读课本内容

信息社会责任：
- 优秀：讨论AI伦理边界，分析冲突，制定保护公约
- 基础：了解网络安全，不泄露个人信息
- 缺失：仅学习软件操作技巧

【输出】每个素养含：分数、匹配度计算过程、匹配证据、理由

输出JSON：
{{
"avg_score": "四素养平均分",
"各素养": {{
  "信息意识": {{"分数":85,"相似度计算过程":{{...}},"匹配证据":["..."],"理由":"..."}},
  "计算思维": {{...}},
  "数字化学习与创新": {{...}},
  "信息社会责任": {{...}}
}}
}}

教案：{text}
"""
    return model_invocation(prompt)


def score_lesson_plan(diagnosis_result):
    """按权重计算三个维度的总分"""
    try:
        def safe_get_score(data, *keys, default=0):
            try:
                for key in keys:
                    if not isinstance(data, dict):
                        return default
                    data = data.get(key, {})
                if isinstance(data, str):
                    import re
                    numbers = re.findall(r'\d+\.?\d*', data)
                    return float(numbers[0]) if numbers else default
                return float(data) if data is not None else default
            except:
                return default

        section_score = safe_get_score(diagnosis_result, "环节完整性", "score")
        time_score = safe_get_score(diagnosis_result, "时间分配", "score")
        literacy_score = safe_get_score(diagnosis_result, "素养匹配", "avg_score")

        total_score = section_score * 0.3 + time_score * 0.3 + literacy_score * 0.4
        return round(total_score, 2)
    except Exception as e:
        st.error(f"评分计算失败: {str(e)}")
        return 0.0


# ---------------------- 4. 图形化界面（上传+显示结果）----------------------
def Main_interface():
    st.set_page_config(page_title="智能诊断系统", layout="wide")
    st.title("📚 初中信息科技教案智能诊断系统")
    st.write("上传TXT教案文件，自动检测规范性并生成建议！")

    File_Upload = st.file_uploader("选择教案文件（TXT格式）", type=["txt"], key="uploader")

    if File_Upload:
        try:
            text = File_Upload.read().decode("utf-8")
        except:
            st.error("文件编码错误！请用记事本打开教案，另存为UTF-8格式TXT")
            return

        if len(text.strip()) < 100:
            st.warning("教案内容过短，可能无法准确检测！")
            return

        # 调用检测功能（添加错误处理）
        with st.spinner("大模型分析中...（约需15-20秒，分三步检测）"):
            try:
                result = test_lesson_plan(text)
            except Exception as e:
                st.error(f"诊断过程出错：{str(e)}")
                return

        # 显示结果
        st.subheader("📊 诊断结果（JSON格式）")
        st.json(result)

        # 计算总分
        total_score = score_lesson_plan(result)

        # 安全提取各维度分数
        section_score = result.get("环节完整性", {}).get("score", 0) or 0
        time_score = result.get("时间分配", {}).get("score", 0) or 0
        literacy_score = result.get("素养匹配", {}).get("avg_score", 0) or 0

        # 显示分数卡片
        st.subheader("📈 教案评分")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            label="环节完整性",
            value=f"{section_score}分",
            help="满分100分，评估导入/新授/练习/总结/作业5环节是否齐全有效"
        )
        col2.metric(
            label="时间分配",
            value=f"{time_score}分",
            help="满分100分，评估各环节时长是否符合45分钟课程要求"
        )
        col3.metric(
            label="核心素养",
            value=f"{literacy_score}分",
            help="满分100分，评估信息意识/计算思维等四大素养覆盖度"
        )
        col4.metric(
            label="🏆 教案总分",
            value=f"{total_score}分",
            help="加权计算：环节30% + 时间30% + 素养40%"
        )

        st.progress(int(total_score), text=f"教案综合评分：{total_score}/100分")

        # 生成文字建议
        suggestions_prompt = f"根据诊断结果{result}，给老师写3条具体修改建议（简洁明了，用1. 2. 3.编号）："

        try:
            response = client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": suggestions_prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            text_suggestions = response.choices[0].message.content
        except Exception as e:
            text_suggestions = f"生成建议时出错：{str(e)}"

        st.subheader("💡 老师修改建议")
        st.info(text_suggestions)


# ---------------------- 运行界面 ----------------------
if __name__ == "__main__":
    Main_interface()
