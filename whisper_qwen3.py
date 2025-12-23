import toml
import os
import pprint
import torch
from tqdm import tqdm
import time
import pysubs2
import re
from pathlib import Path

# 加载配置
config = toml.load('local_whisper_config.toml')
work_dir = config['work_dir']
export_dir = config['export_dir']
file_type = config['file_type']
language = config['language']
model_size = config['model_size']
initial_prompt = config['initial_prompt']

# 修改：使用 get() 方法提供默认值，并处理字符串类型的布尔值
export_srt = config.get('export_srt', True)
if isinstance(export_srt, str):
    export_srt = export_srt.lower() in ['yes', 'true', '1']

if_translate = config.get('if_translate', True)
if isinstance(if_translate, str):
    if_translate = if_translate.lower() in ['yes', 'true', '1']

target_language = config.get('target_language', '简体中文')

if_summary = config.get('if_summary', False)
if isinstance(if_summary, str):
    if_summary = if_summary.lower() in ['yes', 'true', '1']

is_split = config.get('is_split', False)
if isinstance(is_split, str):
    is_split = is_split.lower() in ['yes', 'true', '1']

split_method = config.get('split_method', 'Modest')
sub_style = config.get('sub_style', 'default')

is_vad_filter = config.get('is_vad_filter', True)
if isinstance(is_vad_filter, str):
    is_vad_filter = is_vad_filter.lower() in ['yes', 'true', '1']

set_beam_size = config.get('set_beam_size', 5)

# LLM配置
llm_model_name = config.get('llm_model_name', 'Qwen/Qwen3-0.6B')

# 生成配置
translation_config = config.get('translation_config', {})
summary_config = config.get('summary_config', {})

# 处理过程
my_root_name = work_dir.split('/')[-1]
media_names = []
for root, d_names, f_names in os.walk(work_dir):
    folders = root.split('/')
    for folder in folders:
        if folder.startswith('.'):
            continue
    for d_name in list(d_names):  # 修改：使用 list() 创建副本
        if d_name.startswith('.'):
            d_names.remove(d_name)
    for f_name in f_names:
        if f_name.lower().endswith(
            ('mp3', 'm4a', 'flac', 'aac', 'wav', 'mp4', 'mkv', 'ts', 'flv')):
            media_names.append(f_name)

if not os.path.exists(export_dir):
    os.makedirs(export_dir)

pprint.pprint(media_names)
print("待处理文件数：", len(media_names))

if len(media_names) == 0:
    print("错误：未找到媒体文件，请检查 work_dir 配置")
    exit(1)

choice = input("请检查待处理文件是否正确，若错误请重新检查配置（y/n）\nPlease verify if the files to be processed are correct. If incorrect, please recheck the configuration (y/n): ")
if choice.lower() != "y":
    exit()

# 处理环节
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './temp/hf-cache'
from faster_whisper import WhisperModel


def split_text(text, max_word_count):
    def count_words(text):
        words = re.findall(r'\b\w+\b', text)
        return len(words)

    sentences = re.split(r'(?<=[,.])\s', text)
    new_paragraphs = []
    current_paragraph = ''
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = count_words(sentence)
        if current_word_count + sentence_word_count <= max_word_count:
            current_paragraph += sentence + ' '
            current_word_count += sentence_word_count
        else:
            if current_word_count > 0:
                new_paragraphs.append(current_paragraph.strip())
            current_paragraph = sentence + ' '
            current_word_count = sentence_word_count

    if current_paragraph != '':
        new_paragraphs.append(current_paragraph.strip())

    return new_paragraphs


print('开始转录，请等待...')

file_names = media_names
file_basenames = []
for i in range(len(file_names)):
    file_basenames.append(Path(file_names[i]).stem)
output_dir = Path(export_dir).parent.resolve()

for i in range(len(file_names)):
    torch.cuda.empty_cache()
    
    # 初始化 Whisper 模型（支持本地路径）
    print(f"Loading Whisper model: {model_size}")
    try:
        whisper_model = WhisperModel(
            model_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "int8"
        )
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("请检查模型路径是否正确，或者检查网络连接")
        exit(1)
    
    torch.cuda.empty_cache()
    
    file_name = file_names[i]
    file_basename = file_basenames[i]
    
    if file_type == "video":
        print('提取音频中 Extracting audio from video file...')
        input_path = Path(work_dir) / file_name
        output_audio = f'{file_basename}.mp3'
        os.system(
            f'ffmpeg -i "{input_path}" -f mp3 -ab 192000 -vn "{output_audio}"')
        print('提取完毕 Done.')

    tic = time.time()
    print('识别中 Transcribe in progress...')

    audio_path = Path(work_dir) / file_name
    segments, info = whisper_model.transcribe(
        audio=str(audio_path),
        beam_size=set_beam_size,
        language=language,
        vad_filter=is_vad_filter,
        initial_prompt=initial_prompt,
        vad_parameters=dict(min_silence_duration_ms=1000))

    total_duration = round(info.duration, 2)
    results = []
    pure_texts = []

    with tqdm(total=total_duration, unit=" seconds") as pbar:
        for s in segments:
            segment_dict = {'start': s.start, 'end': s.end, 'text': s.text}
            results.append(segment_dict)
            if language == 'zh':
                if not s.text.endswith(tuple([',', '.', '，', '。'])):
                    pure_texts.append(s.text + ',')
                else:
                    pure_texts.append(s.text)
            else:
                pure_texts.append(s.text)
            segment_duration = s.end - s.start
            pbar.update(segment_duration)
    full_text = ''.join(pure_texts)

    toc = time.time()
    print('识别完毕 Done')
    print(f'Time consumption {toc-tic:.2f}s')
    del whisper_model
    torch.cuda.empty_cache()

    # 初始化 Qwen3 模型（如果需要翻译或摘要）
    if if_translate or if_summary:
        from whispertranslator.qwen3 import Qwen3, GenerationConfig
        
        # 使用配置文件中的模型路径
        print(f"Loading LLM model: {llm_model_name}")
        try:
            qwen3_model = Qwen3(model_path=llm_model_name)
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            print("将跳过翻译和摘要步骤")
            if_translate = False
            if_summary = False
        else:
            # 翻译任务使用非思考模式配置（更高效）
            translate_gen_config = GenerationConfig(
                temperature=translation_config.get('temperature', 0.7),
                top_p=translation_config.get('top_p', 0.8),
                top_k=translation_config.get('top_k', 20),
                max_new_tokens=translation_config.get('max_new_tokens', 2048),
                presence_penalty=translation_config.get('presence_penalty', 1.5)
            )
            
            # 摘要任务可以启用思考模式获得更好的结果
            summary_gen_config = GenerationConfig(
                temperature=summary_config.get('temperature', 0.6),
                top_p=summary_config.get('top_p', 0.95),
                top_k=summary_config.get('top_k', 20),
                max_new_tokens=summary_config.get('max_new_tokens', 4096),
                presence_penalty=summary_config.get('presence_penalty', 1.5)
            )
            
            translator_system_prompt = f"你是一个专业的翻译助手。把下列文字翻译成{target_language}，修改和补充语序让它更符合{target_language}习惯，只返回翻译结果，不要添加任何解释。"
            
            summary_system_prompt = f"你是一个专业的摘要助手。用{target_language}总结下列文字的主题，要求简洁明了。"

    # 获取翻译文本用于字幕
    if if_translate:
        translate_results = []
        print("开始翻译字幕...")
        for idx, segment in enumerate(tqdm(results, desc="翻译进度")):
            try:
                # 使用 /no_think 标记确保不启用思考模式（提高翻译效率）
                translate_text = qwen3_model.infer(
                    translator_system_prompt,
                    segment['text'].replace(' ', '') + " /no_think",
                    translate_gen_config,
                    enable_thinking=False
                )
                
                # 清理翻译结果
                translated_content = translate_text.text.strip().replace('\n', ' ')
                
                translate_segment_dict = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'] + r"\\N" + translated_content
                }
                translate_results.append(translate_segment_dict)
            except Exception as e:
                print(f"\n翻译第 {idx+1} 段时出错: {e}")
                # 出错时保留原文
                translate_segment_dict = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text']
                }
                translate_results.append(translate_segment_dict)

    # 保存完整文本
    new_paragraphs = split_text(full_text, max_word_count=200)
    chunk_filename = file_basename + '.txt'
    chunk_filename = Path(export_dir) / chunk_filename
    
    with open(chunk_filename, 'w', encoding='utf-8') as file:
        for chunk in new_paragraphs:
            file.write(chunk + '\n')

    # 翻译完整文本
    if if_translate:
        translate_filename = file_basename + '_translate' + '.txt'
        translate_filename = Path(export_dir) / translate_filename
        
        print("开始翻译完整文本...")
        with open(translate_filename, 'w', encoding='utf-8') as file:
            for chunk in tqdm(new_paragraphs, desc="翻译文本段落"):
                chunk = chunk.replace("\n", ".")
                try:
                    chunk_translate = qwen3_model.infer(
                        translator_system_prompt,
                        f"{chunk} /no_think",
                        translate_gen_config,
                        enable_thinking=False
                    )
                    
                    # 去除空格和清理结果
                    translated_text = chunk_translate.text.replace(" ", "").strip()
                    
                    # 检查重复（简单的启发式检查）
                    if len(translated_text) > 4:
                        last_4_chars = translated_text[-4:]
                        if translated_text.count(last_4_chars) > 10:
                            print("\n检测到重复，重新生成...")
                            chunk_translate = qwen3_model.infer(
                                translator_system_prompt,
                                f"{chunk} /no_think",
                                translate_gen_config,
                                enable_thinking=False
                            )
                            translated_text = chunk_translate.text.replace(" ", "").strip()
                    
                    print(f"\n原文: {chunk[:50]}...")
                    print(f"译文: {translated_text[:50]}...")
                    file.write(translated_text + '\n')
                    
                except Exception as e:
                    print(f"\n翻译段落时出错: {e}")
                    file.write(chunk + '\n')  # 出错时保留原文

    # 保存字幕文件
    subs = pysubs2.load_from_whisper(results)
    srt_filename = file_basename + '.srt'
    srt_filename = Path(export_dir) / srt_filename
    subs.save(srt_filename)

    if if_translate:
        translate_subs = pysubs2.load_from_whisper(translate_results)
        translate_srt_filename = file_basename + '_translate' + '.srt'
        translate_srt_filename = Path(export_dir) / translate_srt_filename
        translate_subs.save(translate_srt_filename)

    # 转换为ASS格式
    try:
        from srt2ass import srt2ass
        ass_filename = srt2ass(str(srt_filename), sub_style, is_split, split_method)
        print('ASS subtitle saved as: ' + ass_filename)

        if if_translate:
            translate_ass_filename = srt2ass(str(translate_srt_filename),
                                             sub_style, is_split, split_method)
            print('Translated ASS subtitle saved as: ' + translate_ass_filename)
    except ImportError:
        print("Warning: srt2ass module not found, skipping ASS conversion")
    except Exception as e:
        print(f"Error converting to ASS: {e}")

    print('文件字幕生成完毕 / file(s) was completed!')
    
    # 生成摘要
    if if_summary:
        print("开始生成摘要...")
        summary_source_file = translate_filename if if_translate else chunk_filename
        
        try:
            with open(summary_source_file, 'r', encoding='utf-8') as file:
                content = file.read()
                # 摘要任务可以使用思考模式获得更好的效果
                summary_result = qwen3_model.infer(
                    summary_system_prompt,
                    str(content).replace(' ', '').replace('\n', '')[:4000],  # 限制输入长度
                    summary_gen_config,
                    enable_thinking=True  # 摘要任务启用思考模式
                )
                summary_text = summary_result.text
                
                print("总结结果：", summary_text)
                if summary_result.thinking_content:
                    print("思考过程：", summary_result.thinking_content[:200] + "...")
                
                content = summary_text + '\n\n' + content
            
            with open(summary_source_file, "w", encoding='utf-8') as file:
                file.write(content)
        except Exception as e:
            print(f"生成摘要时出错: {e}")
    
    # 清理资源
    import gc
    if if_translate or if_summary:
        if 'qwen3_model' in locals():
            del qwen3_model
    gc.collect()
    torch.cuda.empty_cache()

print('所有字幕生成完毕 All done!')
