import { Worker } from "bullmq";
import IORedis from "ioredis";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { QdrantVectorStore } from "@langchain/qdrant";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import dotenv from "dotenv";
import axios from "axios";
import fs from "fs";
import path from "path";

dotenv.config({ path: "../.env" });

// 🔌 Redis connection
const connection = new IORedis(process.env.REDIS_URL, {
  maxRetriesPerRequest: null,
  tls: {}, // Upstash requires TLS
});

const worker = new Worker(
  "file-upload-queue",
  async (job) => {
    console.log(`📄 Processing job:`, job.data);

    try {
      const fileUrl = job.data.url;
      if (!fileUrl) throw new Error("No file URL provided in job data");

      // Download PDF from Cloudinary
      const response = await axios.get(fileUrl, { responseType: "arraybuffer" });
      const tempFilePath = path.join("/tmp", path.basename(fileUrl));
      fs.writeFileSync(tempFilePath, response.data);
      console.log(`✅ PDF downloaded to temp path: ${tempFilePath}`);

      // Load PDF
      const loader = new PDFLoader(tempFilePath);
      const docs = await loader.load();
      console.log(`✅ Loaded ${docs.length} docs from PDF`);

      if (!docs.length) {
        console.error("⚠️ No text extracted from PDF");
        return;
      }

      // Split docs into chunks
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });
      const splitDocs = await splitter.splitDocuments(docs);
      console.log(`✂️ Split into ${splitDocs.length} chunks`);

      // Embeddings model
      const embeddings = new HuggingFaceTransformersEmbeddings({
        modelName: "Xenova/all-MiniLM-L6-v2",
      });

      // Connect to Qdrant
      try {
        console.log("🔌 Connecting to Qdrant:", process.env.QDRANT_URL);
        const vectorStore = await QdrantVectorStore.fromExistingCollection(
          embeddings,
          {
            url: process.env.QDRANT_URL || "http://localhost:6333",
            apiKey: process.env.QDRANT_API_KEY,
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

      // Cleanup temp file
      fs.unlinkSync(tempFilePath);
      console.log("🧹 Temp PDF file deleted");
    } catch (error) {
      console.error("❌ Error processing PDF:", error);
    }
  },
  { connection } // pass IORedis instance
);

// Test Redis connectivity
connection.ping().then((res) => console.log("✅ Redis PING:", res));
