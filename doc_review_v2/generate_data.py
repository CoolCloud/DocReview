"""
公文核校训练数据生成 V2
采用 JSONL 格式 + 更友好的数据结构
参考最佳实践：每行一个 JSON 对象，包含完整的错误标注信息
"""
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path


# 扩展的错误模式库
ERROR_PATTERNS = {
    # 1. 的地得混用（高频错误）
    "de_di_de": {
        "description": "的地得混用",
        "examples": [
            {"correct": "走的很快", "wrong": "走地很快"},
            {"correct": "缓慢地前进", "wrong": "缓慢的前进"},
            {"correct": "说得很好", "wrong": "说的很好"},
            {"correct": "跑得快", "wrong": "跑的快"},
            {"correct": "做得好", "wrong": "做的好"},
            {"correct": "认真的态度", "wrong": "认真地态度"},
            {"correct": "坚定地推进", "wrong": "坚定的推进"},
            {"correct": "有效的方法", "wrong": "有效地方法"},
            {"correct": "积极地响应", "wrong": "积极的响应"},
            {"correct": "努力地工作", "wrong": "努力的工作"},
        ]
    },
    
    # 2. 同音字错误
    "homophones": {
        "description": "同音字混淆",
        "examples": [
            {"correct": "制定计划", "wrong": "制订计划"},
            {"correct": "截止日期", "wrong": "截至日期"},
            {"correct": "反映问题", "wrong": "反应问题"},
            {"correct": "必须完成", "wrong": "必需完成"},
            {"correct": "启用新系统", "wrong": "起用新系统"},
            {"correct": "做客", "wrong": "作客"},
            {"correct": "权利", "wrong": "权力"},
            {"correct": "申报", "wrong": "申报"},
            {"correct": "启事", "wrong": "起事"},
            {"correct": "账号", "wrong": "帐号"},
            {"correct": "部署工作", "wrong": "布署工作"},
            {"correct": "符合要求", "wrong": "复合要求"},
        ]
    },
    
    # 3. 标点符号错误
    "punctuation": {
        "description": "标点符号使用不当",
        "examples": [
            {"correct": "通知如下：", "wrong": "通知如下。"},
            {"correct": "第一、第二", "wrong": "第一，第二"},
            {"correct": "要求如下。", "wrong": "要求如下："},
            {"correct": "。特此通知", "wrong": "，特此通知"},
            {"correct": "研究决定，", "wrong": "研究决定。"},
            {"correct": "有以下几点：", "wrong": "有以下几点，"},
        ]
    },
    
    # 4. 量词搭配错误
    "measure_words": {
        "description": "量词搭配不当",
        "examples": [
            {"correct": "一份文件", "wrong": "一个文件"},
            {"correct": "三项措施", "wrong": "三个措施"},
            {"correct": "两条建议", "wrong": "两个建议"},
            {"correct": "五个部门", "wrong": "五家部门"},
        ]
    },
    
    # 5. 书面语错误
    "formal_language": {
        "description": "非正式用语",
        "examples": [
            {"correct": "关于", "wrong": "关与"},
            {"correct": "贯彻", "wrong": "贯切"},
            {"correct": "截止", "wrong": "截至"},
            {"correct": "鉴于", "wrong": "监于"},
            {"correct": "拟于", "wrong": "你于"},
        ]
    },
    
    # 6. 常见错别字
    "typos": {
        "description": "常见错别字",
        "examples": [
            {"correct": "取得", "wrong": "取的"},
            {"correct": "进一步", "wrong": "近一步"},
            {"correct": "落实", "wrong": "落石"},
            {"correct": "认真", "wrong": "任真"},
            {"correct": "精神", "wrong": "经神"},
            {"correct": "组织", "wrong": "祖织"},
            {"correct": "具体", "wrong": "据体"},
            {"correct": "确保", "wrong": "确报"},
        ]
    }
}


# 公文模板库（更加丰富）
DOCUMENT_TEMPLATES = [
    # 通知类
    "根据{org}的指示精神，现将有关事项通知如下。",
    "经{org}研究决定，同意{action}。",
    "现将《{doc}》印发给你们，请认真贯彻执行。",
    "为做好{topic}工作，现通知如下。",
    "接上级通知，定于{date}举办{activity}，请各单位积极配合。",
    
    # 报告类
    "根据工作安排，现将{topic}情况报告如下。",
    "{org}高度重视{issue}，多次召开专题会议研究部署。",
    "经过{time}的努力，{work}取得了显著成效。",
    "通过{activity}，有效提升了{result}。",
    
    # 请示类
    "关于{topic}的请示收悉，经研究，现答复如下。",
    "为{purpose}，特请示如下。",
    "鉴于{reason}，拟于{date}开展{activity}。",
    
    # 决定类
    "经{org}研究，决定{decision}。",
    "为进一步{purpose}，特制定本办法。",
    "根据《{law}》的规定，结合实际情况，制定本方案。",
    
    # 函类
    "贵单位关于{topic}的函收悉，经研究，现函复如下。",
    "因{reason}，特致函贵单位，请予以支持配合。",
    
    # 纪要类
    "{org}于{date}召开了{meeting}，会议听取了{report}。",
    "会议认为，{opinion}。会议决定，{decision}。",
    "会议要求，各部门要{requirement}，确保{goal}。",
    
    # 意见类
    "为贯彻落实{policy}，提出如下意见。",
    "关于{topic}工作，提出以下指导意见。",
    
    # 总结类
    "本年度，{org}围绕{goal}，扎实开展各项工作。",
    "一年来，在{support}的支持下，{achievement}。",
    "{activity}圆满完成，达到了预期目标。",
    
    # 批复类
    "你单位《关于{topic}的请示》收悉，经研究，批复如下。",
    "经审核，{opinion}，同意{decision}。",
    
    # 公告类
    "根据{basis}，经{org}研究决定，现将{content}公告如下。",
    "为{purpose}，特此公告。",
]


# 数据字典
ORGANIZATIONS = [
    "上级部门", "本局", "市政府", "教育部", "财政部", "人事处", 
    "办公室", "省委", "区政府", "发展改革委", "工信局"
]

ACTIONS = [
    "开展此项工作", "实施该方案", "进行整改", "推进项目", 
    "组织活动", "加强管理", "启动调研", "落实政策"
]

TOPICS = [
    "项目申报", "经费使用", "人员调配", "设备采购", 
    "安全生产", "环境保护", "教育改革", "民生工程",
    "疫情防控", "党建工作", "干部培训", "绩效考核"
]

DATES = ["2024年3月15日", "2025年6月1日", "2026年1月10日", "近期", "下月初"]

ACTIVITIES = ["培训", "检查", "评估", "调研", "座谈会", "研讨会", "专项行动"]

DOCUMENTS = ["实施方案", "管理办法", "工作细则", "通知要求", "指导意见", "工作规范"]


def generate_document_text() -> str:
    """生成一个公文段落"""
    template = random.choice(DOCUMENT_TEMPLATES)
    
    # 填充模板
    text = template.format(
        org=random.choice(ORGANIZATIONS),
        action=random.choice(ACTIONS),
        topic=random.choice(TOPICS),
        date=random.choice(DATES),
        activity=random.choice(ACTIVITIES),
        doc=random.choice(DOCUMENTS),
        issue=random.choice(TOPICS),
        time="两个月",
        work=random.choice(ACTIONS),
        result="工作质量和效率",
        purpose=random.choice(["加强管理", "提高效率", "规范流程", "落实政策"]),
        reason="工作需要",
        law="中华人民共和国" + random.choice(["行政法", "劳动法", "教育法"]),
        meeting=random.choice(["专题会议", "工作会议", "座谈会"]),
        report="有关情况汇报",
        opinion="当前形势总体向好",
        decision=random.choice(["启动该项目", "实施新政策", "调整组织架构"]),
        requirement="认真贯彻落实",
        goal="工作顺利推进",
        policy="上级文件精神",
        support="上级部门",
        achievement="各项工作取得新进展",
        basis="相关法律法规",
        content="有关事项"
    )
    
    return text


def inject_errors(text: str, error_rate: float = 0.7) -> Tuple[str, List[Dict]]:
    """
    在文本中注入错误，返回错误文本和错误详情
    error_rate: 错误注入概率（提高到 70%）
    """
    # 决定是否注入错误
    if random.random() > error_rate:
        return text, []
    
    errors = []
    chars = list(text)
    current_text = "".join(chars)
    
    # 收集所有可能的错误注入点
    injection_candidates = []
    
    for pattern_key, pattern in ERROR_PATTERNS.items():
        for error_pair in pattern["examples"]:
            correct = error_pair["correct"]
            wrong = error_pair["wrong"]
            
            # 查找所有匹配位置
            pos = current_text.find(correct)
            if pos != -1:
                injection_candidates.append({
                    "position": pos,
                    "correct": correct,
                    "wrong": wrong,
                    "pattern_key": pattern_key,
                    "pattern_desc": pattern["description"]
                })
    
    # 如果找到可注入的位置，随机选择 1-2 个
    if injection_candidates:
        num_errors = min(random.randint(1, 2), len(injection_candidates))
        selected = random.sample(injection_candidates, num_errors)
        
        # 按位置排序（从后往前替换，避免位置偏移）
        selected.sort(key=lambda x: -x["position"])
        
        for candidate in selected:
            pos = candidate["position"]
            correct = candidate["correct"]
            wrong = candidate["wrong"]
            
            # 记录错误信息
            errors.append({
                "position": pos,
                "length": len(correct),
                "correct_text": correct,
                "wrong_text": wrong,
                "error_type": candidate["pattern_key"],
                "error_desc": candidate["pattern_desc"]
            })
            
            # 替换为错误文本
            chars[pos:pos+len(correct)] = list(wrong)
    
    return "".join(chars), errors


def create_training_sample(original_text: str, error_text: str, errors: List[Dict]) -> Dict:
    """
    创建训练样本（新格式）
    使用更友好的数据结构
    """
    # 创建字符级别的标签（用于 Token Classification）
    labels = ["O"] * len(error_text)
    
    for error in errors:
        start = error["position"]
        end = start + len(error["wrong_text"])
        
        if start < len(labels):
            labels[start] = "B-ERROR"
            for i in range(start + 1, min(end, len(labels))):
                labels[i] = "I-ERROR"
    
    return {
        "id": None,  # 将在保存时分配
        "text": error_text,
        "original_text": original_text,
        "labels": labels,
        "errors": errors,
        "error_count": len(errors),
        "has_error": len(errors) > 0,
        "error_types": list(set(e["error_type"] for e in errors)) if errors else []
    }


def generate_dataset(num_samples: int, desc: str = "数据") -> List[Dict]:
    """生成数据集"""
    dataset = []
    
    print(f"正在生成 {num_samples} 条{desc}...")
    
    for i in range(num_samples):
        # 生成原始文本
        original_text = generate_document_text()
        
        # 注入错误（70% 概率）
        error_text, errors = inject_errors(original_text, error_rate=0.7)
        
        # 创建样本
        sample = create_training_sample(original_text, error_text, errors)
        sample["id"] = i + 1
        dataset.append(sample)
        
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i + 1}/{num_samples} 条")
    
    return dataset


def save_jsonl(data: List[Dict], filepath: str):
    """保存为 JSONL 格式（每行一个 JSON）"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: List[Dict], filepath: str):
    """保存为标准 JSON 格式（用于查看）"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_statistics(dataset: List[Dict], name: str):
    """打印数据集统计信息"""
    total = len(dataset)
    with_errors = sum(1 for d in dataset if d["has_error"])
    without_errors = total - with_errors
    
    # 统计错误类型
    error_type_counts = {}
    for sample in dataset:
        for error_type in sample.get("error_types", []):
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    print(f"\n{name}统计:")
    print(f"  总样本数: {total}")
    print(f"  含错误: {with_errors} ({with_errors/total*100:.1f}%)")
    print(f"  无错误: {without_errors} ({without_errors/total*100:.1f}%)")
    
    if error_type_counts:
        print(f"  错误类型分布:")
        for error_type, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
            desc = ERROR_PATTERNS[error_type]["description"]
            print(f"    - {desc} ({error_type}): {count}")


def main():
    """主函数"""
    print("="*70)
    print("公文核校训练数据生成 V2")
    print("格式: JSONL (每行一个JSON) + 清晰的错误标注")
    print("="*70)
    
    # 创建数据目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 生成数据（增加到 1000 条总数据）
    print("\n生成训练和测试数据...")
    train_data = generate_dataset(800, "训练数据")
    test_data = generate_dataset(200, "测试数据")
    
    # 保存为 JSONL 格式（推荐格式，便于流式读取和追加）
    print("\n保存数据...")
    save_jsonl(train_data, "data/train.jsonl")
    save_jsonl(test_data, "data/test.jsonl")
    
    # 同时保存 JSON 格式（便于人工查看）
    save_json(train_data[:10], "data/train_sample.json")  # 保存前10条样本
    save_json(test_data[:5], "data/test_sample.json")
    
    # 打印统计信息
    print("\n" + "="*70)
    print("数据生成完成！")
    print("="*70)
    
    print_statistics(train_data, "训练集")
    print_statistics(test_data, "测试集")
    
    print("\n文件保存位置:")
    print(f"  训练集 (JSONL): data/train.jsonl")
    print(f"  测试集 (JSONL): data/test.jsonl")
    print(f"  训练样本 (JSON): data/train_sample.json (前10条)")
    print(f"  测试样本 (JSON): data/test_sample.json (前5条)")
    
    # 显示几个样本
    print("\n" + "="*70)
    print("样本展示:")
    print("="*70)
    
    for i, sample in enumerate(train_data[:3], 1):
        print(f"\n【样本 {i}】")
        print(f"ID: {sample['id']}")
        print(f"原文: {sample['original_text']}")
        print(f"错误文本: {sample['text']}")
        print(f"是否有错: {sample['has_error']}")
        print(f"错误数量: {sample['error_count']}")
        
        if sample['errors']:
            print(f"错误详情:")
            for err in sample['errors']:
                print(f"  - 位置 {err['position']}: '{err['correct_text']}' -> '{err['wrong_text']}'")
                print(f"    类型: {err['error_desc']} ({err['error_type']})")
        print("-" * 70)


if __name__ == "__main__":
    main()
