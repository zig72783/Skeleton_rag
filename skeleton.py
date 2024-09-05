import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration
import fitz  # PyMuPDF
#import faiss
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#import faiss.contrib.torch_utils

class TextChunker:
    def __init__(self, max_chunk_length, chunk_overlap, min_chunk_length=1):
        self.max_chunk_length = max_chunk_length
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def chunk_text(self, text):
        chunks = []
        start = 0
        text_length = len(text)     
        while start < text_length:
            end = min(start + self.max_chunk_length, text_length)
            chunk = text[start:end]
            chunks.append(chunk)
            if end == text_length:
                break
            start += self.max_chunk_length - self.chunk_overlap      
        # 检查最后一个 chunk 是否小于 min_chunk_length，并决定是否合并
        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_length:
            last_chunk = chunks.pop()
            chunks[-1] += last_chunk
        return chunks

class TextProcessor:
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
        self.model = model

    def process(self, chunks):
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda:3')
        outputs = self.model(**inputs)
        return outputs

class SummaryGenerator:
    def __init__(self, model_name):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to('cuda:3')
        self.max_summary_length = 300  # 最大摘要长度

    def generate_summary(self, chunk):
        prompt = "<s>摘要：</s>\n\n" + chunk  # 添加生成摘要的提示
        max_length = self.tokenizer.model_max_length if self.tokenizer.model_max_length > 0 else 512 # 获取模型的最大长度，如果没有预定义最大长度，设置为默认值512
        # 使用 tokenizer 对带有 prompt 的 chunk 进行编码，并指定 max_length
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to('cuda:3')
        # 生成摘要
        summary_ids = self.model.generate(
            inputs.input_ids,
            max_length=self.max_summary_length,
            min_length=10,
            length_penalty=1.0,
            num_beams=4,
            no_repeat_ngram_size=3,  # 避免重复生成相同的内容
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True) # 解码生成的 token IDs 为文本
        return summary

class Skeleton:
    def __init__(self, processor_model, summary_model_name, chunk_size, num_layer):
        self.chunk_size = chunk_size
        self.num_layer = num_layer
        self.processor = TextProcessor(processor_model)
        self.summary_generator = SummaryGenerator(summary_model_name)
        #self.vector_database = faiss.IndexFlatL2(768)
        #self.vector_database = [faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 768)]
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/sentence-t5-large")
        self.vector_database = {}
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.layer_vector_databases = {}

    def process_and_summarize(self, chunks):
        summaries = self.generate_summaries(chunks)
        return summaries

    def tokenize_chunks(self, chunks):
        token_ids = []
        for chunk in chunks:
            ids = self.tokenizer.encode(chunk, truncation=True, padding='max_length', max_length=self.chunk_size)
            token_ids.append(ids)
        return token_ids

    def generate_summaries(self, chunks):
        summary = []
        summaries = []
        for chunk in chunks:
            summary = self.summary_generator.generate_summary(chunk)
            summaries.append(summary)
        return summaries

    def build_layers(self, summaries): 
        layer_vector_databases = {}    
        for j in range(self.num_layer):
            chunks = chunker.chunk_text(''.join(summaries))
            summaries = skeleton.process_and_summarize(chunks)     
            #layer_summary = ''.join(summaries)                    # 检查是否实现分层summary
            #print(f"   ***Layer {j+1}***:\n {layer_summary}")     # 检查是否实现分层summary          
            vector_database = FAISS.from_texts(summaries, embeddings)
            layer_vector_databases[j] = vector_database
            #vector = embeddings.embed_query(summaries[0])                 # 获取摘要的嵌入向量
            #print(f"Vector_database {j+1} - Vector shape: {len(vector)}") # 获取摘要的嵌入向量和向量维度
        return layer_vector_databases

    def retrieve_from_layers(self, query, k=3):
        all_retrieved_summaries = []
        for j in range(self.num_layer):
            # 从当前层的数据库中检索最相关的 k 个摘要
            vector_database = layer_vector_databases[j]
            retrieved_summaries = vector_database.similarity_search(query, k=k)
            # 打印每层的检索结果
            print(f"*** Layer {j+1} retrieved summaries: ***")
            for idx, summary in enumerate(retrieved_summaries):
                print(f"  Summary {idx+1}: {summary.page_content}")
            all_retrieved_summaries.extend(retrieved_summaries)   # 将当前层检索出的摘要加入最终结果

        return all_retrieved_summaries

#    def answer_question(self, query):
#        relevant_summaries = self.retrieve_relevant_summaries(query)
#        combined_text = ' '.join(relevant_summaries) + ' ' + query
#        answer = self.summary_generator.generate_summary(combined_text)
#        print(f"The answer is: {answer}")
#        return answer

# 加载处理器模型
processor_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto"
)

# 加载embedding模型
model_name = "sentence-transformers/sentence-t5-large"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 初始化 Skeleton
skeleton = Skeleton(processor_model, "facebook/bart-large-cnn", 512, 3)

# 加载文本
def get_pdf_text_from_url(url):
    response = requests.get(url)
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)
    pdf_document = fitz.open('temp.pdf')
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

query = "What is HMT?"
input_url = "https://arxiv.org/pdf/2405.06067"
text = get_pdf_text_from_url(input_url)
chunker = TextChunker(max_chunk_length=500, chunk_overlap=20) # 初始化分块器
chunks = chunker.chunk_text(text) # 调用chunk_text分块
summaries = skeleton.process_and_summarize(chunks) #调用骨架进行分层总结
layer_vector_databases = skeleton.build_layers(summaries) # 构建分层向量数据库
all_retrieved_summaries = skeleton.retrieve_from_layers(query, k=3) #创建召回

# 输出摘要
#for i, summary in enumerate(summaries):
    #print(f"Summary {i+1}: {summary}")

# 打印最终检索到的摘要
#for idx, summary in enumerate(all_retrieved_summaries):
#    print(f"Final Summary {idx+1}: {summary.page_content}")

#输出回答
#print(answer)