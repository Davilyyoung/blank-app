import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 设置API
os.environ["OPENAI_API_KEY"] = "sk-5e030cc687b846718c190775c9ac6064"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"

DATA_DIR = "data/contracts"
VECTOR_DIR = "vectorstores/contracts"

st.set_page_config(
    page_title="合同智能问答系统",
    layout="wide"
)

# ================= 工具函数 =================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents():
    documents = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return documents

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        try:
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs = loader.load()
            elif file.lower().endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
            else:
                continue

            for d in docs:
                d.metadata["source"] = file
                documents.append(d)
        except Exception as e:
            st.error(f"加载 {file} 失败：{e}")

    return documents


def build_vectorstore():
    docs = load_documents()
    if not docs:
        return False, "未找到合同文件"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    split_docs = splitter.split_documents(docs)

    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    os.makedirs(os.path.dirname(VECTOR_DIR), exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)

    return True, f"向量库构建完成（{len(split_docs)} 个片段）"


def ask_llm(question: str):
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    if not docs:
        return "未找到相关内容。", []

    context = "\n\n".join(
        f"来源：{d.metadata.get('source')}\n内容：{d.page_content}"
        for d in docs
    )

    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.3,
        max_tokens=1500
    )

    prompt = ChatPromptTemplate.from_template(
        """你是一名严谨的法律助理，请严格依据给定的合同内容回答问题。
如果合同中没有明确说明，请回答“合同中未明确约定”。

【合同内容】
{context}

【问题】
{question}

【要求】
- 只基于合同内容回答
- 用清晰、条理化的中文
- 不要编造合同中不存在的条款
"""
    )

    chain = prompt | llm
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer.content.strip(), docs


# ================= 页面 =================

st.title("📄 合同智能问答系统")

with st.sidebar:
    st.header("📂 合同管理")

    if st.button("🔄 构建 / 更新向量库"):
        with st.spinner("正在构建向量库..."):
            ok, msg = build_vectorstore()
        if ok:
            st.success(msg)
        else:
            st.warning(msg)

    st.markdown("---")
    st.markdown("**使用说明**")
    st.markdown(
        """
        1. 将合同 PDF / TXT 放入 `data/contracts`
        2. 点击「构建向量库」
        3. 在右侧输入合同问题
        """
    )

# 主区域
question = st.text_input("请输入合同问题：", placeholder="例如：合同的违约责任是什么？")

if st.button("🔍 查询") and question:
    if not os.path.exists(VECTOR_DIR):
        st.warning("请先构建向量库")
    else:
        with st.spinner("正在分析合同内容..."):
            answer, docs = ask_llm(question)

        st.subheader("✅ 综合答案")
        st.write(answer)

        with st.expander("📄 查看引用的合同原文"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**段落 {i}｜来源：{d.metadata.get('source')}**")
                st.write(d.page_content)
                st.markdown("---")
