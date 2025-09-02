import express from "express";
import cors from "cors";
import multer from "multer";
import { Queue } from "bullmq";
import axios from "axios"; // Using axios for API calls
import { QdrantVectorStore } from "@langchain/qdrant";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import dotenv from "dotenv";
dotenv.config({ path: "../.env" });



// Load environment variables from .env file

const app = express();
app.use(cors());
app.use(express.json());

// -------------------------
// BullMQ queue (Redis)
// -------------------------
// const fileQueue = new Queue("file-upload-queue", {
//   connection: {
//     host: "localhost",
//     port: 6379,
//   },
// });

const fileQueue = new Queue("file-upload-queue", {
  connection: { url: process.env.REDIS_URL },
});

// -------------------------
// Multer storage for file uploads
// -------------------------
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + "-" + Math.round(Math.random() * 1e9);
    cb(null, file.originalname + "-" + uniqueSuffix);
  },
});

const upload = multer({ storage });

// -------------------------
// Routes
// -------------------------

// Health check
app.get("/", (req, res) => {
  return res.json({ status: "All good" });
});

// Upload PDF
app.post("/upload/pdf", upload.single("pdf"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  // Add job to BullMQ
  await fileQueue.add("file-ready", {
    filename: req.file.originalname,
    destination: req.file.destination,
    path: req.file.path,
  });

  return res.status(200).json({
    message: "File uploaded successfully",
    file: {
      name: req.file.originalname,
      path: req.file.path,
    },
  });
});

// Chat endpoint
app.post("/chat", async (req, res) => {
  
  const { query } = req.body;
  console.log(process.env.DEEPSEEK_API_KEY);
  if (!query) {
    return res.status(400).json({ error: "Query is required" });
  }

  try {
    // Use a free local embedding model (no API key needed)
    const embeddings = new HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });

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
      You are a helpful AI Agent. Use the provided documents to answer user queries.dont say in answer about document you get answer be professional
      Context, also dont answer question not in docs, only you can answer from docs and basic formal replies: ${docs.map(d => d.pageContent).join('\n')}
    `;

    // Call DeepSeek via OpenRouter with proper authentication
    const response = await axios.post(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        model: "deepseek/deepseek-chat", // Make sure this is the correct model name
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: query },
        ],
      },
      {
        headers: {
          "Authorization": `Bearer ${process.env.DEEPSEEK_API_KEY}`, // Your OpenRouter API key
          "HTTP-Referer": process.env.SITE_URL || "http://localhost:3000", // Required by OpenRouter
          "X-Title": process.env.APP_NAME || "RAG Application", // Optional but recommended
          "Content-Type": "application/json"
        }
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
      details: error.response?.data || error.message 
    });
  }
});

// -------------------------
// Start server
// -------------------------
app.listen(8000, () => console.log(`✅ Server started at PORT: 8000`));
