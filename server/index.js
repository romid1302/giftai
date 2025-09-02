import express from "express";
import cors from "cors";
import multer from "multer";
import { Queue } from "bullmq";
import axios from "axios";
import { QdrantVectorStore } from "@langchain/qdrant";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import dotenv from "dotenv";
import { v2 as cloudinary } from "cloudinary";
import "./worker.js"; // worker runs separately
dotenv.config({ path: "../.env" });

const app = express();
app.use(cors());
app.use(express.json());

// -------------------------
// BullMQ queue (Redis)
// -------------------------
const fileQueue = new Queue("file-upload-queue", {
  connection: { url: process.env.REDIS_URL },
});

// -------------------------
// Cloudinary config
// -------------------------
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// -------------------------
// Multer storage for temp uploads
// -------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, file.originalname + "-" + uniqueSuffix);
  },
});
const upload = multer({ storage });

// -------------------------
// Routes
// -------------------------

// Health check
app.get("/", (req, res) => res.json({ status: "All good" }));

// Upload PDF
app.post("/upload/pdf", upload.single("pdf"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  try {
    // Upload PDF to Cloudinary
    const result = await cloudinary.uploader.upload(req.file.path, {
      resource_type: "raw", // important for PDFs
      folder: "pdfs",
    });

    // Add job to BullMQ with Cloudinary URL
    await fileQueue.add("file-ready", {
      filename: req.file.originalname,
      url: result.secure_url,
    });

    return res.status(200).json({
      message: "File uploaded successfully",
      file: {
        name: req.file.originalname,
        url: result.secure_url,
      },
    });
  } catch (err) {
    console.error("Cloudinary upload error:", err);
    return res.status(500).json({ error: "Failed to upload file" });
  }
});

// Chat endpoint
app.post("/chat", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ error: "Query is required" });

  try {
    // Embedding model
    const embeddings = new HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });

    // Connect to Qdrant
    const vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddings,
      {
        url: process.env.QDRANT_URL || "http://localhost:6333",
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: "pdf-docs",
      }
    );

    const retriever = vectorStore.asRetriever({ k: 2 });
    const docs = await retriever.invoke(query);

    const SYSTEM_PROMPT = `
      You are a helpful AI Agent. Use the provided documents to answer user queries.
      Do not mention document sources in your answer.
      Context: ${docs.map(d => d.pageContent).join("\n")}
    `;

    // Call DeepSeek via OpenRouter
    const response = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "deepseek/deepseek-chat",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: query },
        ],
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.DEEPSEEK_API_KEY}`,
          "HTTP-Referer": process.env.SITE_URL || "http://localhost:3000",
          "X-Title": process.env.APP_NAME || "RAG Application",
          "Content-Type": "application/json",
        },
      }
    );

    return res.json({
      answer: response.data.choices[0].message.content,
      docs: docs.map(d => ({ pageContent: d.pageContent })),
    });
  } catch (error) {
    console.error("Error in chat endpoint:", error.response?.data || error.message);
    return res.status(500).json({
      error: "Internal server error",
      details: error.response?.data || error.message,
    });
  }
});

// -------------------------
// Start server
// -------------------------
app.listen(8000, () => console.log(`✅ Server started at PORT: 8000`));
