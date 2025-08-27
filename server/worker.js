import { Worker } from "bullmq";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
// Or use HuggingFaceInferenceEmbeddings if you want API-hosted models
import { QdrantVectorStore } from "@langchain/qdrant";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";

dotenv.config({ path: "../.env" });

// Redis connection (Upstash)
const connection = {
  url: process.env.REDIS_URL, // e.g. rediss://default:xxx@upstash-url:6379
};

const worker = new Worker(
  "file-upload-queue",
  async (job) => {
    console.log(`📄 Processing job:`, job.data);

    try {
      // Load PDF
      const loader = new PDFLoader(job.data.path);
      const docs = await loader.load();
      console.log(`✅ Loaded ${docs.length} docs from PDF`);

      if (!docs.length) {
        console.error("⚠️ No text extracted from PDF");
        return;
      }

      // Split docs into chunks for better embeddings
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const splitDocs = await splitter.splitDocuments(docs);
      console.log(`✂️ Split into ${splitDocs.length} chunks`);

      // Embeddings model
      const embeddings = new HuggingFaceTransformersEmbeddings({
        modelName: "Xenova/all-MiniLM-L6-v2", // Local CPU model
      });

      try {
        // Connect to Qdrant (Cloud or local)
        const vectorStore = await QdrantVectorStore.fromExistingCollection(
          embeddings,
          {
            url: process.env.QDRANT_URL || "http://localhost:6333",
            apiKey: process.env.QDRANT_API_KEY, // needed for cloud
            collectionName: "pdf-docs",
          }
        );

        await vectorStore.addDocuments(splitDocs);
        console.log(`✅ Added ${splitDocs.length} chunks to existing collection`);
      } catch (err) {
        console.log("⚠️ Collection not found, creating new...");
        await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
          url: process.env.QDRANT_URL || "http://localhost:6333",
          apiKey: process.env.QDRANT_API_KEY,
          collectionName: "pdf-docs",
        });
        console.log(`✅ Created collection and stored ${splitDocs.length} chunks`);
      }
    } catch (error) {
      console.error("❌ Error processing PDF:", error);
    }
  },
  { connection }
);
