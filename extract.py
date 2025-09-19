#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理Kvasir Polyp数据集，从p0-p9字段提取所有句子并生成结构化文本
"""

import langextract as lx
import json
import os
import time
from typing import List, Dict, Any

def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {file_path}，包含 {len(data)} 个样本")
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return []

def extract_all_prompts(sample: Dict[str, Any]) -> str:
    """从样本中提取p0-p9的所有句子"""
    try:
        prompts = sample.get('prompts', {})
        all_sentences = []
        
        # 遍历p0到p9
        for i in range(10):
            prompt_key = f'p{i}'
            if prompt_key in prompts:
                prompt_list = prompts[prompt_key]
                if isinstance(prompt_list, list):
                    # 添加该prompt下的所有句子
                    all_sentences.extend(prompt_list)
                elif isinstance(prompt_list, str):
                    # 如果是字符串，直接添加
                    all_sentences.append(prompt_list)
        
        # 将所有句子合并为一个文本，用句号分隔
        return '. '.join(sentence.strip() for sentence in all_sentences if sentence.strip())
        
    except Exception as e:
        print(f"提取所有prompts时出错: {e}")
        return ""

def setup_langextract():
    """设置LangExtract的提示和示例"""
    
    # 定义 extraction prompt
    prompt_description = """
    Extract key information useful for medical image segmentation from clinical text describing polyps. 
    For each described region, extract and group the following attributes: region_entity, location, size, color, and shape.  
    
    Guidelines:
    1. region_entity refers to the main described object (e.g., polyp, lesion, tumor, infection, nodule, etc.).  
    2. If an attribute is missing, output "none".  
    3. Correct typos and normalize terms if possible.  
    4. If multiple regions are described, assign each region a unique group identifier (e.g., region_1, region_2, …).  
    5. Ensure that attributes belonging to the same region share the same group identifier.  
    """
    
    # 定义示例
    examples = [
        lx.data.ExampleData(
            text="one medium yellow round polyp which is a projecting growth of tissue located in right of the image",
            extractions=[
                lx.data.Extraction(
                    extraction_class="region_entity", extraction_text="polyp", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="size", extraction_text="medium", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="color", extraction_text="yellow", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="shape", extraction_text="round", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="location", extraction_text="right of the image", attributes={"region_group": "region_1"}
                )
            ]
        ),
        lx.data.ExampleData(
            text="two small white oval polyp which is a small lump in the lining of colon located in top left, bottom right of the image",
            extractions=[
                lx.data.Extraction(
                    extraction_class="region_entity", extraction_text="polyp", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="size", extraction_text="small", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="color", extraction_text="white", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="shape", extraction_text="oval", attributes={"region_group": "region_1"}
                ),
                lx.data.Extraction(
                    extraction_class="location", extraction_text="top left, bottom right of the image", attributes={"region_group": "region_1"}
                )
            ]
        )
    ]
    
    return prompt_description, examples

def extract_structured_text(input_text: str, prompt_description: str, examples: List, api_key: str, max_retries: int = 5) -> str:
    """使用LangExtract进行结构化文本抽取，带重试机制"""
    if not input_text.strip():
        return "No text to extract"
    
    for attempt in range(max_retries):
        try:
            print(f"  尝试第 {attempt + 1} 次抽取...", end="")
            
            # 调用 LangExtract
            result = lx.extract(
                text_or_documents=input_text,
                prompt_description=prompt_description,
                examples=examples,
                api_key=api_key,
                model_id="gemini-2.5-flash",
            )
            
            # 整理结果为结构化文本
            regions = {}
            for extraction in result.extractions:
                group = extraction.attributes.get("region_group", "ungrouped")
                if group not in regions:
                    regions[group] = {}
                regions[group][extraction.extraction_class] = extraction.extraction_text
            
            # 生成结构化文本
            structured_parts = []
            for region, attrs in regions.items():
                region_parts = []
                
                # 按固定顺序输出属性
                for attr_name in ["region_entity", "size", "color", "shape", "location"]:
                    if attr_name in attrs:
                        region_parts.append(f"{attr_name}: {attrs[attr_name]}")
                    else:
                        region_parts.append(f"{attr_name}: none")
                
                if region_parts:
                    structured_parts.append(f"[{region}] " + ", ".join(region_parts))
            
            print(" 成功!")
            return " | ".join(structured_parts) if structured_parts else "No structured information extracted"
            
        except Exception as e:
            print(f" 失败: {str(e)[:100]}...")
            
            # 如果是配额错误，等待更长时间
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                wait_time = min(120 * (2 ** attempt), 600)  # 指数退避，最大10分钟
                print(f"  配额限制，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                # 其他错误，等待较长时间
                wait_time = min(15 * (attempt + 1), 60)  # 线性增长，最大60秒
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
            # 如果是最后一次尝试，返回错误信息
            if attempt == max_retries - 1:
                return f"Error after {max_retries} attempts: {str(e)}"
    
    return f"Failed after {max_retries} attempts"

def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
    """加载已有的处理结果，支持断点续传，并移除失败的结果"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # 移除失败的结果（包含"Error after"的结果）
            successful_results = []
            failed_count = 0
            
            for result in existing_results:
                text = result.get('text', '')
                if 'Error after' in text or 'Failed after' in text:
                    failed_count += 1
                    print(f"移除失败的样本: {result.get('segment_id', 'unknown')}")
                else:
                    successful_results.append(result)
            
            print(f"发现已有结果文件，包含 {len(existing_results)} 个样本")
            print(f"成功的样本: {len(successful_results)} 个")
            print(f"失败的样本: {failed_count} 个（将重新处理）")
            
            return successful_results
        except Exception as e:
            print(f"加载已有结果文件时出错: {e}")
            return []
    return []

def get_processed_segment_ids(results: List[Dict[str, Any]]) -> set:
    """获取已处理的segment_id集合"""
    return {result.get('segment_id', '') for result in results}

def process_dataset():
    """处理Kvasir Polyp数据集"""
    
    # 配置参数
    data_dir = './data/original_text_data'
    api_key = 'YOUR_GOOGLE_API_KEY'
    output_file = './structured_text_results.json'
    
    print("开始处理Kvasir Polyp数据集...")
    print(f"数据目录: {data_dir}")
    print(f"输出文件: {output_file}")
    
    # 设置LangExtract
    prompt_description, examples = setup_langextract()
    
    # 要处理的文件
    json_files = ['train.json', 'val.json', 'test.json']
    
    # 加载已有结果，支持断点续传
    all_results = load_existing_results(output_file)
    processed_ids = get_processed_segment_ids(all_results)
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        print(f"\n处理文件: {json_file}")
        
        # 加载数据
        samples = load_json_file(file_path)
        
        # 处理每个样本
        for i, sample in enumerate(samples):
            segment_id = sample.get('segment_id', '')
            
            # 跳过已处理的样本
            if segment_id in processed_ids:
                print(f"\n跳过已处理样本 {i+1}/{len(samples)}: {segment_id}")
                continue
                
            print(f"\n处理样本 {i+1}/{len(samples)}: {segment_id}")
            
            # 提取基本信息
            img_name = sample.get('img_name', '')
            mask_name = sample.get('mask_name', '')
            
            # 提取p0-p9的所有句子
            input_text = extract_all_prompts(sample)
            
            if not input_text:
                print(f"  警告: 样本 {segment_id} 没有prompts文本")
                structured_text = "No input text available"
            else:
                print(f"  原始文本: {input_text[:80]}...")
                # 进行结构化文本抽取（带重试机制）
                structured_text = extract_structured_text(input_text, prompt_description, examples, api_key, max_retries=5)
                print(f"  结构化文本: {structured_text}")
            
            # 保存结果
            result = {
                'segment_id': segment_id,
                'img_name': img_name,
                'mask_name': mask_name,
                'text': structured_text  # 使用'text'字段名，符合需求
            }
            
            all_results.append(result)
            
            # 实时保存结果，防止API限额导致数据丢失
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                print(f"  已保存 {len(all_results)} 个结果到文件")
            except Exception as e:
                print(f"  保存文件时出错: {e}")
            
            # 在每个样本处理完成后添加更长间隔，避免请求过于频繁
            if i < len(samples) - 1:  # 不是最后一个样本
                print("  等待 10 秒后处理下一个样本...")
                time.sleep(10)
        
        print(f"\n完成处理 {json_file}: {len(samples)} 个样本")
    
    # 保存所有结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n所有结果已保存到: {output_file}")
        print(f"总共处理了 {len(all_results)} 个样本")
        
        # 显示一些统计信息
        print("\n处理完成!")
        print("输出格式: segment_id, img_name, mask_name, text")
        
    except Exception as e:
        print(f"保存结果时出错: {e}")

if __name__ == "__main__":
    process_dataset()
