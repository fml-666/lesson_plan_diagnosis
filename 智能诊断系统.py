# 教案诊断系统.py
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
    """调用大模型检测3个维度"""
    prompt = f"""
你是初中信息科技教案诊断专家，请先深呼吸，仔细阅读完整教案，再进行分析上传的教案文件，输出3部分结果：

1. 教学环节完整性（总分100分，100=完全合格）
   检查项：必须包含【导入、新授、练习、总结、作业】5个环节，每个环节需满足：  
   存在性：环节名称明确（如"导入""新授"等，含变体及格式变体：导入/引入/情境导入、新授/新课/知识讲解、练习/实践/探究活动、总结/小结/课堂总结、作业/作业布置/课后作业/五、作业/六、作业等，均映射到标准环节，分散在教案各处的同类活动应合并计算）；  
   内容有效性：环节描述≥30字，（包括该环节下所有子活动和具体指令），且含具体操作（如"导入用微信红包案例提问"而非"导入新课"）。  
   评分逻辑：  
   每缺失1个环节扣20分（5环节全缺失得0分）；  
   每有1个环节内容无效（字数不足或无具体操作）扣10分；  
   扣完为止，最低0分。
   【强制要求】
   - 必须逐段扫描全文，不能漏检
   - "作业"环节特别注意：搜索"作业"、"课后"、"任务"、"布置"等关键词
   - 分散活动合并：同一环节的多个活动累加计算字数
   输出字段：  
   'score'：最终得分（0-100）；  
   '详情'：文字描述扣分原因（如"缺失'作业'环节，扣20分；'练习'环节内容仅20字，扣10分"）；  
   '缺失环节'：仅列出得0分的环节名称，缺失的标准环节名称列表（如["作业", "总结"]）；  
   '各环节状态'：每个环节的状态列表，含：  
     '环节'：标准环节名（如"导入"）；  
     '得分'：20/10/0
     '是否存在'：true/false；  
     '内容有效'：true/false（仅当存在时判断，字数≥30字且有具体操作）；  
     '摘要'：环节核心内容概括（≤50字）。

2. 时间分配合理性（总分100分，100=完全合理）
   【强制一致性规则】：必须严格基于第1部分"环节完整性"的判定结果
   - 若第1部分某环节"是否存在"=true且"得分">0，则本部分该环节必须"当前时长">0，"是否合理"按实际时长判断
   - 若第1部分某环节"得分"=0（不存在），则本部分"当前时长"=0，"建议时长"="环节缺失"，"是否合理"=false
   - 禁止重新判断环节存在性，必须复用第1部分结论

   检查项：总课时40分钟，各环节参考时长：  
   导入：3-5分钟（±2分钟）→ 合理范围1-7分钟
   新授：15-20分钟（±5分钟）→ 合理范围10-25分钟  
   练习：8-15分钟（±3分钟）→ 合理范围5-18分钟
   总结：2-4分钟（±1分钟）→ 合理范围1-5分钟
   作业：2-3分钟（±1分钟）→ 合理范围1-4分钟。  

   评分逻辑：  
   每有1个存在环节时长超出参考范围扣10分；  
   总时长（各环节之和）超出45±5分钟（40-50分钟）扣15分；  
   扣完为止，最低0分。 

   输出字段：  
   'score'：最终得分（0-100）；  
   '详情'：具体扣分说明，如"'新授'环节时长28分钟超出25分钟范围（扣10分），总时长50分钟超出45分钟（扣15分），总分75分"；
   '建议'：每个环节的建议列表，含：  
     '环节'：标准环节名；  
     '当前时长'：从教案中提取的实际分钟数（若第1部分判定缺失则为0）；
     '建议时长'：参考范围（如"3-7分钟"）或"环节缺失"；  
     '是否合理'：true/false（仅当第1部分判定存在且时长在范围内时为true）。

3. 核心素养匹配度（基于参照案例的语义相似度加权计算）
   【算法说明】
   采用"锚点对比法"计算语义相似度：
   - 每个素养设置3个参照文本（锚点）：优秀Case（90-100分档）、基础Case（60-80分档）、缺失Case（0-30分档）
   - 将用户教案与3个锚点进行语义对比，计算匹配度权重
   - 最终得分 = 优秀锚点匹配度×0.5 + 基础锚点匹配度×0.3 + （1-缺失锚点匹配度）×0.2
   - 结果映射到0-100分区间，保留两位小数

   【四维素养参照锚点】（少样本学习示例）

   【信息意识】
   - 优秀锚点（95分特征）："通过'网络谣言辨一辨'活动，学生需对比官方媒体与自媒体信息，验证来源真实性，并记录判断依据"
   - 基础锚点（70分特征）："引导学生辨别网络信息真伪，培养信息甄别意识"  
   - 缺失锚点（15分特征）："学习使用搜索引擎"（仅技能操作，无素养培养）

   【计算思维】
   - 优秀锚点（95分特征）："将垃圾分类任务分解为'识别-判断-归类'三步骤，绘制流程图，并用自然语言描述算法逻辑"
   - 基础锚点（70分特征）："按照步骤完成操作，理解问题解决的过程"
   - 缺失锚点（15分特征）："记住操作步骤"（仅记忆，无思维过程）

   【数字化学习与创新】
   - 优秀锚点（95分特征）："小组协作使用在线文档共同编辑研究报告，插入多媒体素材，并进行在线演示分享"
   - 基础锚点（70分特征）："利用Word制作电子小报，整合网络资料"
   - 缺失锚点（15分特征）："阅读课本内容"（无数字化工具应用）

   【信息社会责任】
   - 优秀锚点（95分特征）："讨论AI换脸技术的伦理边界，分析'convenience vs privacy'的冲突，制定个人信息保护公约"
   - 基础锚点（70分特征）："了解网络安全知识，不泄露个人信息"
   - 缺失锚点（15分特征）："学习软件操作技巧"（无伦理讨论）

   【评分计算逻辑】
   对每个素养执行以下分析步骤：

   Step 1：匹配度评估（在0-1之间，保留两位小数）
   - 与优秀锚点的语义相似度：____（如0.85）
   - 与基础锚点的语义相似度：____（如0.60）  
   - 与缺失锚点的语义相似度：____（如0.10）

   Step 2：加权计算（权重：优秀0.5，基础0.3，缺失反向0.2）
   原始分 = 0.85×0.5 + 0.60×0.3 + (1-0.10)×0.2 = 0.425 + 0.18 + 0.18 = 0.785
   映射到百分制：0.785 × 100 = 78.50分

   Step 3：证据提取（必须列出匹配关键词）
   - 优秀锚点匹配词："对比信息来源"、"验证真实性"（来自教案第X段）
   - 基础锚点匹配词："辨别真伪"（来自教案目标描述）
   - 缺失锚点差异：教案未提及"记录判断依据"（优秀锚点特有）

   【重要】以下仅为格式示例，实际分析时必须：
1. 通读完整教案，识别所有分散的教学活动
2. 将分散的同类活动（如多个"练习"）合并为一个环节统计
3. 根据实际内容生成摘要，不要复制示例文字
4. 字数统计包含该环节下所有具体描述

输出严格JSON格式（仅格式示例，内容必须基于实际教案分析）：
{{
"环节完整性": {{
    "score": "0-100, 根据实际环节计算：5环节各20分，缺失0分，简略10分，完整20分",
    "详情": "文字描述，说明各环节得分计算过程",
    "缺失环节": "仅列出得0分的环节，如无则留空数组",
    "各环节状态": [
        {{
            "环节": "导入", 
            "得分": "0-20, 20=完整有效，10=存在但简略，0=缺失",
            "是否存在": "true/false",
            "内容有效": "true/false, 仅存在时判断，字数≥30且有具体操作",
            "摘要": "该环节实际内容概括，≤50字"
        }},
        {{
            "环节": "新授",
            "得分": "0-20",
            "是否存在": "true/false",
            "内容有效": "true/false",
            "摘要": "..."
        }},
        {{
            "环节": "练习",
            "得分": "0-20",
            "是否存在": "true/false",
            "内容有效": "true/false",
            "摘要": "..."
        }},
        {{
            "环节": "总结",
            "得分": "0-20",
            "是否存在": "true/false",
            "内容有效": "true/false",
            "摘要": "..."
        }},
        {{
            "环节": "作业",
            "得分": "0-20",
            "是否存在": "true/false",
            "内容有效": "true/false",
            "摘要": "..."
        }}
    ]
}},
"时间分配": {{
    "score": "0-100, 根据实际时长计算：每超范围扣10分，总时长超40-50分钟扣15分",
    "详情": "文字描述，说明各时间扣分情况",
    "建议": [
        {{
            "环节": "导入",
            "当前时长": "0或具体分钟数, 第1部分判定缺失则为0",
            "建议时长": "X-Y分钟或环节缺失, 缺失时标注环节缺失",
            "是否合理": "true/false, 仅当存在且时长在范围内为true"
        }},
        {{"环节": "新授", "当前时长": "0或数字", "建议时长": "...", "是否合理": "true/false"}},
        {{"环节": "练习", "当前时长": "0或数字", "建议时长": "...", "是否合理": "true/false"}},
        {{"环节": "总结", "当前时长": "0或数字", "建议时长": "...", "是否合理": "true/false"}},
        {{"环节": "作业", "当前时长": "0或数字", "建议时长": "...", "是否合理": "true/false"}}
    ]
}},
"素养匹配": {{
    "avg_score": "0-100, 四素养平均分，保留两位小数",
    "各素养": {{
        "信息意识": {{
            "分数": "0-100",
            "相似度计算过程": {{
                "优秀锚点匹配度": "0.00-1.00",
                "基础锚点匹配度": "0.00-1.00",
                "缺失锚点匹配度": "0.00-1.00",
                "权重计算公式": "优秀×0.5 + 基础×0.3 + (1-缺失)×0.2 = 结果 → 百分制"
            }},
            "匹配证据": ["引用教案原文片段，说明匹配依据"],
            "理由": "综合评分说明"
        }},
        "计算思维": {{
            "分数": "0-100",
            "相似度计算过程": "...",
            "匹配证据": ["..."],
            "理由": "..."
        }},
        "数字化学习与创新": {{
            "分数": "0-100",
            "相似度计算过程": "...",
            "匹配证据": ["..."],
            "理由": "..."
        }},
        "信息社会责任": {{
            "分数": "0-100",
            "相似度计算过程": "...",
            "匹配证据": ["..."],
            "理由": "..."
        }}
    }}
}}
}}

教案文本：{text}
"""
    return model_invocation(prompt)


def score_lesson_plan(diagnosis_result):
    """按权重计算三个维度的总分"""
    try:
        # 安全获取分数值，并转换为数字（处理字符串情况）
        def safe_get_score(data, *keys, default=0):
            """安全获取嵌套字典中的分数，并转换为float"""
            try:
                for key in keys:
                    data = data.get(key, {})
                # 如果是字符串，尝试提取数字
                if isinstance(data, str):
                    # 提取字符串中的第一个数字
                    import re
                    numbers = re.findall(r'\d+\.?\d*', data)
                    if numbers:
                        return float(numbers[0])
                    return default
                # 如果是数字，直接返回
                return float(data) if data is not None else default
            except:
                return default

        section_score = safe_get_score(diagnosis_result, "环节完整性", "score")
        time_score = safe_get_score(diagnosis_result, "时间分配", "score")
        literacy_score = safe_get_score(diagnosis_result, "素养匹配", "avg_score")

        # 加权计算
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

    # 上传文件（仅支持TXT，避免PDF乱码）
    File_Upload = st.file_uploader("选择教案文件（TXT格式）", type=["txt"], key="uploader")

    if File_Upload:
        # 读取TXT内容（UTF-8编码防乱码）
        try:
            text = File_Upload.read().decode("utf-8")
        except:
            st.error("文件编码错误！请用记事本打开教案，另存为UTF-8格式TXT")
            return

        if len(text.strip()) < 100:
            st.warning("教案内容过短，可能无法准确检测！")
            return

        # 调用检测功能
        with st.spinner("大模型分析中...（约10秒）"):
            result = test_lesson_plan(text)

        # 显示结果
        st.subheader("📊 诊断结果（JSON格式）")
        st.json(result)  # 结构化展示
        # 计算总分
        total_score = score_lesson_plan(result)
        # 提取各维度分数
        section_score = result.get("环节完整性", {}).get("score", 0)
        time_score = result.get("时间分配", {}).get("score", 0)
        literacy_score = result.get("素养匹配", {}).get("avg_score", 0)
        # 显示分数卡片
        st.subheader("📈 教案评分")

        # 创建分数卡片布局
        col1, col2, col3, col4 = st.columns(4)

        # 环节完整性卡片
        col1.metric(
            label="环节完整性",
            value=f"{section_score}分",
            help="满分100分，评估导入/新授/练习/总结/作业5环节是否齐全有效"
        )

        # 时间分配卡片
        col2.metric(
            label="时间分配",
            value=f"{time_score}分",
            help="满分100分，评估各环节时长是否符合45分钟课程要求"
        )

        # 核心素养卡片
        col3.metric(
            label="核心素养",
            value=f"{literacy_score}分",
            help="满分100分，评估信息意识/计算思维等四大素养覆盖度"
        )

        # 总分卡片（突出显示）
        col4.metric(
            label="🏆 教案总分",
            value=f"{total_score}分",
            help="加权计算：环节30% + 时间30% + 素养40%"
        )

        # 添加进度条可视化
        st.progress(int(total_score), text=f"教案综合评分：{total_score}/100分")

        # 生成文字建议（直接调用API，不强制JSON格式）
        suggestions_prompt = f"根据诊断结果{result}，给老师写3条具体修改建议（简洁明了，用1. 2. 3.编号）："

        try:
            # 直接调用API，不经过model_invocation的JSON解析
            response = client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": suggestions_prompt}],
                temperature=0.7,  # 建议部分可以稍微有创意一些
                max_tokens=1000
            )
            text_suggestions = response.choices[0].message.content
        except Exception as e:
            text_suggestions = f"生成建议时出错：{str(e)}"

        st.subheader("💡 老师修改建议")
        st.info(text_suggestions)  # 蓝色背景突出显示


# ---------------------- 运行界面 ----------------------
if __name__ == "__main__":
    Main_interface()  # 启动图形界面