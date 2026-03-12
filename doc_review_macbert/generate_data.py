"""
公文校验训练数据生成器
基于规则生成带有错误标注的训练数据（JSONL 格式）
"""
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path


# 扩展的公文错误模式库
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
            {"correct": "扎实地落实", "wrong": "扎实的落实"},
            {"correct": "深入地研究", "wrong": "深入的研究"},
        ]
    },
    
    # 2. 同音字错误
    "homophones": {
        "description": "同音字混淆",
        "examples": [
            {"correct": "取得成绩", "wrong": "取的成绩"},
            {"correct": "制定计划", "wrong": "制订计划"},
            {"correct": "截止日期", "wrong": "截至日期"},
            {"correct": "反映问题", "wrong": "反应问题"},
            {"correct": "必须完成", "wrong": "必需完成"},
            {"correct": "启用新系统", "wrong": "起用新系统"},
            {"correct": "权利", "wrong": "权力"},
            {"correct": "部署工作", "wrong": "布署工作"},
            {"correct": "符合要求", "wrong": "复合要求"},
            {"correct": "账号", "wrong": "帐号"},
            {"correct": "启事通知", "wrong": "起事通知"},
            {"correct": "关于此事", "wrong": "关与此事"},
        ]
    },
    
    # 3. 标点符号错误
    "punctuation": {
        "description": "标点符号使用不当",
        "examples": [
            {"correct": "根据规定：", "wrong": "根据规定，"},
            {"correct": "第一、第二、第三", "wrong": "第一，第二，第三"},
            {"correct": '关于"通知"的说明', "wrong": "关于'通知'的说明"},
            {"correct": "《办法》规定", "wrong": "<办法>规定"},
            {"correct": "（重要）", "wrong": "【重要】"},
        ]
    },
    
    # 4. 错别字（形近字）
    "typos": {
        "description": "常见错别字",
        "examples": [
            {"correct": "认真负责", "wrong": "任真负责"},
            {"correct": "贯彻落实", "wrong": "贯切落实"},
            {"correct": "水平提高", "wrong": "水准提高"},
            {"correct": "坚持原则", "wrong": "坚特原则"},
            {"correct": "促进发展", "wrong": "促近发展"},
            {"correct": "履行职责", "wrong": "旅行职责"},
            {"correct": "奖惩制度", "wrong": "奖成制度"},
            {"correct": "协调各方", "wrong": "协凋各方"},
        ]
    },
    
    # 5. 非正式用语
    "informal": {
        "description": "非正式用语",
        "examples": [
            {"correct": "各位同志", "wrong": "大家"},
            {"correct": "此致敬礼", "wrong": "谢谢"},
            {"correct": "请予审批", "wrong": "请批准一下"},
            {"correct": "现将有关事项通知如下", "wrong": "现在通知大家"},
            {"correct": "请认真贯彻执行", "wrong": "请好好做"},
        ]
    },
    
    # 6. 词语搭配错误
    "collocation": {
        "description": "词语搭配不当",
        "examples": [
            {"correct": "提高质量", "wrong": "增加质量"},
            {"correct": "增加数量", "wrong": "提高数量"},
            {"correct": "改善条件", "wrong": "提高条件"},
            {"correct": "加强管理", "wrong": "增强管理"},
            {"correct": "增强意识", "wrong": "加强意识"},
            {"correct": "维护权益", "wrong": "保护权益"},
        ]
    }
}


# 正确公文常用句式模板
CORRECT_TEMPLATES = [
    "根据上级文件精神，{action}。",
    "为{purpose}，经研究决定，{action}。",
    "现将{topic}的有关事项通知如下：{content}",
    "各单位要高度重视，{action}，确保{result}。",
    "经研究，同意{topic}，请认真贯彻执行。",
    "{topic}工作开展以来，取得了显著成效。{detail}",
    "为进一步{action}，现提出以下要求：{content}",
    "关于{topic}，经领导批准，{action}。",
    "按照工作部署，{action}，现已{result}。",
    "{department}要切实履行职责，{action}。",
    "会议讨论并通过了{topic}，要求各单位{action}。",
    "为贯彻落实{policy}，{action}，取得实效。",
]


def generate_correct_sentence() -> Tuple[str, List[str]]:
    """生成正确的公文句子"""
    actions = [
        "认真组织学习", "抓好落实工作", "加强监督检查",
        "提高工作效率", "完善相关制度", "做好宣传工作",
        "深入调研论证", "统筹协调推进", "及时报送材料",
        "强化责任担当", "优化工作流程", "创新工作方法"
    ]
    
    purposes = [
        "加强组织建设", "提升服务水平", "推进工作开展",
        "改进工作作风", "规范管理制度", "促进协调发展"
    ]
    
    topics = [
        "会议精神", "工作方案", "管理办法", "实施细则",
        "通知事项", "考核标准", "安全检查", "培训计划"
    ]
    
    contents = [
        "一是强化组织领导；二是明确工作职责；三是加强督促检查",
        "各部门要密切配合，形成工作合力",
        "坚持问题导向，确保工作落到实处",
        "加强统筹协调，提高工作质量和效率"
    ]
    
    results = [
        "各项工作顺利开展", "工作任务按期完成", "取得预期成效",
        "形成良好工作局面", "达到预期目标", "圆满完成任务"
    ]
    
    departments = [
        "各级各部门", "相关职能部门", "各单位各部门",
        "领导小组成员单位", "责任部门"
    ]
    
    policies = [
        "上级文件要求", "会议精神", "工作部署", "相关规定", "决策部署"
    ]
    
    template = random.choice(CORRECT_TEMPLATES)
    text = template.format(
        action=random.choice(actions),
        purpose=random.choice(purposes),
        topic=random.choice(topics),
        content=random.choice(contents),
        result=random.choice(results),
        department=random.choice(departments),
        policy=random.choice(policies),
        detail=random.choice(contents)
    )
    
    # 所有字符标记为 "O" (无错误)
    labels = ["O"] * len(text)
    return text, labels


def inject_error(text: str, labels: List[str]) -> Tuple[str, List[str], bool]:
    """
    在正确文本中注入错误
    
    返回:
        (错误文本, 标签列表, 是否成功注入)
    """
    # 随机选择错误类型
    error_type = random.choice(list(ERROR_PATTERNS.keys()))
    error_examples = ERROR_PATTERNS[error_type]["examples"]
    
    # 尝试注入错误
    random.shuffle(error_examples)  # 随机打乱
    for example in error_examples:
        correct = example["correct"]
        wrong = example["wrong"]
        
        if correct in text:
            # 找到正确词的位置
            pos = text.index(correct)
            
            # 替换为错误词
            new_text = text[:pos] + wrong + text[pos + len(correct):]
            
            # 更新标签（使用 BIO 标注）
            new_labels = labels.copy()
            error_start = pos
            error_end = pos + len(wrong)
            
            # 标注错误位置
            for i in range(error_start, min(error_end, len(new_labels))):
                if i == error_start:
                    new_labels[i] = "B-ERROR"
                else:
                    new_labels[i] = "I-ERROR"
            
            # 如果长度发生变化，调整标签列表
            len_diff = len(wrong) - len(correct)
            if len_diff > 0:
                # 错误词更长，增加标签
                for _ in range(len_diff):
                    new_labels.insert(error_end - 1, "I-ERROR")
            elif len_diff < 0:
                # 错误词更短，删除标签
                for _ in range(-len_diff):
                    if error_end < len(new_labels):
                        new_labels.pop(error_end)
            
            return new_text, new_labels, True
    
    return text, labels, False


def generate_error_sentence() -> Tuple[str, List[str]]:
    """生成包含错误的句子"""
    # 随机选择错误类型和示例
    error_type = random.choice(list(ERROR_PATTERNS.keys()))
    error_example = random.choice(ERROR_PATTERNS[error_type]["examples"])
    
    # 使用错误词构建句子
    wrong_phrase = error_example["wrong"]
    
    # 构建包含错误的公文句子
    templates = [
        f"根据上级文件精神，{wrong_phrase}。",
        f"为加强工作，要求各单位{wrong_phrase}。",
        f"会议强调，各部门要{wrong_phrase}。",
        f"为推进工作，现要求{wrong_phrase}。",
        f"经研究决定，各单位应{wrong_phrase}。",
        f"通知要求，{wrong_phrase}相关工作。",
        f"为确保效果，请{wrong_phrase}。",
        f"各级各部门要{wrong_phrase}，抓好落实。",
    ]
    
    # 随机选择模板
    text = random.choice(templates)
    
    # 查找错误词在句子中的位置
    error_start = text.find(wrong_phrase)
    if error_start == -1:
        # 如果没找到，直接使用错误词作为句子
        text = wrong_phrase + "是工作重点。"
        error_start = 0
    
    # 创建标签
    labels = []
    for i, char in enumerate(text):
        if i >= error_start and i < error_start + len(wrong_phrase):
            if i == error_start:
                labels.append("B-ERROR")
            else:
                labels.append("I-ERROR")
        else:
            labels.append("O")
    
    return text, labels


def generate_dataset(
    num_train: int = 800,
    num_test: int = 200,
    error_rate: float = 0.4,
    output_dir: str = "data"
) -> None:
    """
    生成数据集（JSONL 格式）
    
    参数:
        num_train: 训练集样本数
        num_test: 测试集样本数
        error_rate: 错误样本比例（0-1）
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"📝 开始生成数据集...")
    print(f"   训练集: {num_train} 条 (错误样本约 {int(num_train * error_rate)} 条)")
    print(f"   测试集: {num_test} 条 (错误样本约 {int(num_test * error_rate)} 条)")
    print()
    
    def generate_split(num_samples: int, split_name: str):
        """生成一个数据集分片"""
        data = []
        error_count = 0
        num_errors_target = int(num_samples * error_rate)
        
        print(f"生成 {split_name} 数据...")
        
        for i in range(num_samples):
            # 决定是否生成错误样本
            should_generate_error = (error_count < num_errors_target)
            
            if should_generate_error:
                # 直接生成包含错误的句子
                text, labels = generate_error_sentence()
                has_error = True
                error_count += 1
            else:
                # 生成正确句子
                text, labels = generate_correct_sentence()
                has_error = False
            
            # 创建样本
            sample = {
                "id": f"{split_name}_{i+1}",
                "text": text,
                "labels": labels,
                "has_error": has_error
            }
            data.append(sample)
        
        # 保存为 JSONL
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ {split_name}.jsonl: {len(data)} 条数据, {error_count} 条含错误 ({error_count/len(data)*100:.1f}%)")
        
        # 同时保存为 JSON（用于查看）
        json_file = output_path / f"{split_name}_readable.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data[:10], f, ensure_ascii=False, indent=2)
        print(f"✓ {split_name}_readable.json: 前 10 条样本（可读格式）")
        print()
        
        return data
    
    # 生成训练集和测试集
    train_data = generate_split(num_train, "train")
    test_data = generate_split(num_test, "test")
    
    print("=" * 60)
    print("✅ 数据生成完成！")
    print(f"   输出目录: {output_path.absolute()}")
    print(f"   训练数据: train.jsonl ({num_train} 条)")
    print(f"   测试数据: test.jsonl ({num_test} 条)")
    print("=" * 60)
    print()
    print("📋 数据格式说明:")
    print('   {"id": "train_1", "text": "...", "labels": ["O", "O", ...], "has_error": false}')
    print()
    print("🎯 下一步:")
    print("   python doc_review_macbert/train.py")
    print()


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成公文校验训练数据")
    parser.add_argument("--train", type=int, default=800, help="训练集样本数")
    parser.add_argument("--test", type=int, default=200, help="测试集样本数")
    parser.add_argument("--error-rate", type=float, default=0.4, help="错误样本比例 (0-1)")
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录")
    
    args = parser.parse_args()
    
    generate_dataset(
        num_train=args.train,
        num_test=args.test,
        error_rate=args.error_rate,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
