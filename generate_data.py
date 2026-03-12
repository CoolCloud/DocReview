"""
生成公文核校训练数据
错误类型包括：错别字、标点错误、语法问题等
"""
import json
import random
from typing import List, Tuple

# 常见的公文错误模式
COMMON_ERRORS = {
    # 错别字对
    "的地得": [
        ("走的", "走地"), ("缓慢的", "缓慢地"), ("说的", "说地"),
        ("跑得快", "跑的快"), ("做得好", "做的好")
    ],
    # 标点错误
    "punctuation": [
        ("。", "，"), ("，", "。"), ("、", "，"), ("；", "，")
    ],
    # 同音字错误
    "homophones": [
        ("制定", "制订"), ("启事", "起事"), ("权利", "权力"),
        ("反应", "反映"), ("必须", "必需"), ("截至", "截止"),
        ("做客", "作客"), ("启用", "起用"), ("意义", "意义")
    ],
    # 书写规范
    "format": [
        ("年", "年度"), ("元", "元整"), ("关于", "关与")
    ]
}

# 公文常用句式模板
DOCUMENT_TEMPLATES = [
    "根据{organization}的指示精神，现将有关事项通知如下。",
    "经{organization}研究决定，同意{action}。",
    "为了进一步{purpose}，特制定本办法。",
    "{organization}于{date}召开了{meeting}，会议决定{decision}。",
    "现将{document}印发给你们，请认真贯彻执行。",
    "根据《{law}》的规定，结合实际情况，制定本方案。",
    "经审核，{person}同志符合{condition}，同意{result}。",
    "{organization}高度重视{issue}，要求各部门{action}。",
    "为确保{goal}，现提出以下要求。",
    "特此通知，请遵照执行。",
    "此次{activity}取得了{result}，达到了预期目标。",
    "根据工作需要，决定组织{activity}，请各单位{action}。",
    "{person}同志在{position}期间，工作表现{evaluation}。",
    "经过认真研究，现批复如下。",
    "关于{topic}的请示收悉，经研究，现答复如下。",
]

ORGANIZATIONS = ["上级部门", "本局", "市政府", "教育部", "财政部", "人事处", "办公室"]
ACTIONS = ["开展此项工作", "实施该方案", "进行整改", "推进项目", "组织活动"]
PURPOSES = ["加强管理", "提高效率", "规范流程", "落实政策", "改善服务"]
MEETINGS = ["专题会议", "工作会议", "座谈会", "研讨会", "协调会"]
DECISIONS = ["启动该项目", "实施新政策", "调整组织架构", "加强监督管理"]
DATES = ["2024年3月15日", "2025年6月1日", "2026年1月10日", "上月"]
DOCUMENTS = ["实施方案", "管理办法", "工作细则", "通知要求", "指导意见"]


def inject_error(text: str) -> Tuple[str, List[dict]]:
    """
    在文本中注入错误，返回错误文本和错误标注
    """
    chars = list(text)
    errors = []
    
    # 随机决定是否注入错误（50%概率）
    if random.random() > 0.5:
        error_type = random.choice(list(COMMON_ERRORS.keys()))
        error_pairs = COMMON_ERRORS[error_type]
        
        # 尝试注入1-2个错误
        num_errors = random.randint(1, 2)
        for _ in range(num_errors):
            correct, wrong = random.choice(error_pairs)
            
            # 查找可以替换的位置
            text_str = "".join(chars)
            pos = text_str.find(correct)
            
            if pos != -1:
                # 记录错误位置
                errors.append({
                    "start": pos,
                    "end": pos + len(correct),
                    "correct": correct,
                    "wrong": wrong,
                    "type": error_type
                })
                
                # 替换为错误文本
                for i, c in enumerate(wrong):
                    if pos + i < len(chars):
                        chars[pos + i] = c
                
                # 如果长度不同，需要调整
                if len(wrong) < len(correct):
                    for _ in range(len(correct) - len(wrong)):
                        chars.pop(pos + len(wrong))
                elif len(wrong) > len(correct):
                    for i in range(len(wrong) - len(correct)):
                        chars.insert(pos + len(correct), wrong[len(correct) + i])
    
    return "".join(chars), errors


def generate_document_text() -> str:
    """生成一个公文段落"""
    template = random.choice(DOCUMENT_TEMPLATES)
    
    text = template.format(
        organization=random.choice(ORGANIZATIONS),
        action=random.choice(ACTIONS),
        purpose=random.choice(PURPOSES),
        meeting=random.choice(MEETINGS),
        decision=random.choice(DECISIONS),
        date=random.choice(DATES),
        document=random.choice(DOCUMENTS),
        law="中华人民共和国" + random.choice(["行政法", "劳动法", "教育法"]),
        person=random.choice(["张三", "李四", "王五", "赵六"]),
        condition=random.choice(["任职条件", "晋升要求", "评选标准"]),
        result=random.choice(["批准", "同意", "通过"]),
        issue=random.choice(["安全生产", "环境保护", "教育改革", "民生工程"]),
        goal=random.choice(["工作顺利推进", "任务圆满完成", "目标如期实现"]),
        activity=random.choice(["培训", "检查", "评估", "调研"]),
        evaluation=random.choice(["优秀", "良好", "出色", "突出"]),
        topic=random.choice(["项目申报", "经费使用", "人员调配", "设备采购"]),
        position=random.choice(["科长", "处长", "主任", "经理"])
    )
    
    return text


def create_training_example(text: str, errors: List[dict]) -> dict:
    """
    创建训练样本（序列标注格式）
    使用 BIO 标注：B-ERROR（错误开始）、I-ERROR（错误内部）、O（正常）
    """
    labels = ["O"] * len(text)
    
    for error in errors:
        start, end = error["start"], error["end"]
        if start < len(labels):
            labels[start] = "B-ERROR"
        for i in range(start + 1, min(end, len(labels))):
            labels[i] = "I-ERROR"
    
    return {
        "text": text,
        "labels": labels,
        "errors": errors,
        "has_error": len(errors) > 0
    }


def generate_dataset(num_samples: int = 500) -> List[dict]:
    """生成训练数据集"""
    dataset = []
    
    print(f"正在生成 {num_samples} 条训练数据...")
    
    for i in range(num_samples):
        # 生成原始文本
        original_text = generate_document_text()
        
        # 注入错误
        error_text, errors = inject_error(original_text)
        
        # 创建训练样本
        if errors:  # 有错误的样本
            example = create_training_example(error_text, errors)
            dataset.append(example)
        else:  # 无错误的样本（作为负样本）
            example = create_training_example(original_text, [])
            dataset.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"已生成 {i + 1}/{num_samples} 条数据")
    
    return dataset


def main():
    """生成并保存训练数据"""
    # 生成数据
    train_data = generate_dataset(400)
    test_data = generate_dataset(100)
    
    # 保存数据
    import os
    os.makedirs("data", exist_ok=True)
    
    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open("data/test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    train_errors = sum(1 for d in train_data if d["has_error"])
    test_errors = sum(1 for d in test_data if d["has_error"])
    
    print("\n" + "="*50)
    print("数据生成完成！")
    print("="*50)
    print(f"训练集: {len(train_data)} 条 (含错误: {train_errors})")
    print(f"测试集: {len(test_data)} 条 (含错误: {test_errors})")
    print(f"数据已保存到 data/ 目录")
    
    # 显示示例
    print("\n示例数据:")
    for i, example in enumerate(train_data[:3]):
        print(f"\n样本 {i+1}:")
        print(f"文本: {example['text']}")
        print(f"有错误: {example['has_error']}")
        if example['errors']:
            for err in example['errors']:
                print(f"  错误: {err}")


if __name__ == "__main__":
    main()
