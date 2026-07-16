import express from "express";
import path from "path";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "50mb" }));

// ---------------------------------------------------------------------------
// HARDCODED Q&A — checked BEFORE calling Gemini.
// If the user's message matches one of these, we answer instantly from here
// and never touch the Gemini API for that turn.
// Matching is substring-based (case-insensitive) via `keywords`.
// ---------------------------------------------------------------------------
const HARDCODED_QA = [
  {
    keywords: [
      "campaign",
      "campaign performance",
      "performance",
      "monthly performance",
      "monthly campaigns",
      "campaign report",
      "roas",
      "roas this month",
      "this month performance",
      "marketing performance",
      "account performance",
      "client performance",
      "how are campaigns doing",
      "how did campaigns perform",
      "performance report",
      "monthly report",
    ],
    answer:
      "Blended ROAS across all client accounts rose to 4.2x this month, up from 3.6x last month. Most of that improvement came from Meridian Foods, whose account jumped from 3.1x to 5.4x after we shifted a portion of their Meta budget into lookalike audiences. On the other end, Kavach Security's account declined by 14%, which is the weakest performance in the portfolio this month.",
  },

  {
    keywords: [
      "kavach",
      "kavach account",
      "kavach issue",
      "kavach problem",
      "kavach performance",
      "what happened with kavach",
      "what happened to kavach",
      "tracking issue",
      "tracking problem",
      "tracking broken",
      "form tracking",
      "conversion tracking",
      "landing page",
      "lead tracking",
      "missing leads",
      "lead issue",
      "why is kavach down",
      "kavach decline",
      "kavach drop",
    ],
    answer:
      "Their internal team made a change to the landing page nine days ago, and it appears to have broken the form tracking without anyone noticing. Since then, we have had very limited visibility into how many leads are actually coming through. Based on typical conversion patterns for that account, I would estimate somewhere around forty leads may have gone unrecorded during that window.",
  },

  {
    keywords: [
      "told client",
      "told kavach",
      "informed client",
      "notify client",
      "client informed",
      "did we tell them",
      "did you notify",
      "have we informed",
      "contacted client",
      "emailed client",
      "message client",
      "client update",
    ],
    answer:
      "Not yet. I can prepare a message to their marketing contact if you would like.",
  },

  {
    keywords: [
      "owner",
      "account owner",
      "kavach owner",
      "who owns kavach",
      "who manages",
      "who is managing",
      "account manager",
      "responsible for kavach",
      "assigned to kavach",
      "who handles kavach",
      "account responsibility",
      "ownership",
    ],
    answer:
      "That is actually part of the issue. Kavach's account does not have a designated owner in our system right now. It has been handled informally between you and Priya from the outreach team, which likely explains why this sat unnoticed for over a week.",
  },

  {
    keywords: [
      "rohit",
      "assign rohit",
      "rohit owner",
      "rohit takeover",
      "should rohit manage",
      "make rohit owner",
      "can rohit handle",
      "give rohit",
      "rohit account",
      "rohit responsibility",
      "rohit manage kavach",
    ],
    answer:
      "He could manage the reporting and client check-ins, but he is currently behind on the outreach list for this same account. If he takes on ownership as well, one of those responsibilities is likely to slip.",
  },

  {
    keywords: [
      "reminder",
      "weekly reminder",
      "schedule reminder",
      "set reminder",
      "kavach reminder",
      "weekly check",
      "follow up",
      "follow-up",
      "check in",
      "weekly check-in",
      "monday reminder",
      "schedule check",
    ],
    answer:
      "Done. I have scheduled a weekly check-in for Kavach Security every Monday at ten in the morning, and I have noted in the client file that ownership sits with you until it is formally reassigned.",
  },

  {
    keywords: [
      "unowned accounts",
      "missing owner",
      "accounts without owner",
      "no owner",
      "ownership issue",
      "owner missing",
      "other accounts",
      "similar accounts",
      "who else has no owner",
      "accounts not assigned",
      "unassigned accounts",
      "ownership report",
    ],
    answer:
      "Yes, two others follow a similar pattern. Nandini Textiles and one of the smaller retainer accounts are both being handled informally, without anyone officially responsible for them. Neither has shown a problem yet, but given what just happened with Kavach, I would recommend assigning owners before something similar comes up.",
  },
];

/**
 * Checks the user's latest message against the HARDCODED_QA table.
 * Matching is case-insensitive substring matching against each entry's
 * keyword list. Returns the first matching answer, or null if nothing matches.
 */
function checkHardcodedAnswer(message: string): string | null {
  if (!message || typeof message !== "string") return null;
  const lowerMsg = message.toLowerCase();
  for (const qa of HARDCODED_QA) {
    if (qa.keywords.some((kw) => lowerMsg.includes(kw.toLowerCase()))) {
      return qa.answer;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Gemini client singleton
// ---------------------------------------------------------------------------
let geminiClient: GoogleGenAI | null = null;

/**
 * Lazily instantiates and caches a single GoogleGenAI client using the
 * GEMINI_API_KEY environment variable. Throws clearly if the key is missing,
 * instead of failing later with an opaque auth error.
 */
function getGeminiClient(): GoogleGenAI {
  if (!geminiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error(
        "GEMINI_API_KEY is not set. Add it to your .env file (GEMINI_API_KEY=your_key_here) and restart the server."
      );
    }
    geminiClient = new GoogleGenAI({ apiKey });
  }
  return geminiClient;
}

/**
 * Executes a Gemini API call with automatic retries, exponential backoff,
 * automatic fallback to alternative model aliases if the primary model is
 * unavailable, and a hard timeout so a hung request can never spin forever.
 */
async function generateContentWithRetry(
  client: GoogleGenAI,
  params: { model: string; contents: any; config?: any },
  maxAttempts = 2,
  initialDelayMs = 800,
  timeoutMs = 15000
) {
  // Determine fallback order based on the requested model.
  // gemini-flash-latest tends to be more stable under load than the newer
  // gemini-3.5-flash, so it's tried first.
  const modelQueue: string[] = [params.model];
  if (params.model === "gemini-flash-latest") {
    modelQueue.push("gemini-3.5-flash", "gemini-3.1-flash-lite");
  } else if (params.model === "gemini-3.1-flash-lite-image") {
    modelQueue.push("gemini-3.1-flash-image");
  }

  let lastError: any = null;

  for (const modelName of modelQueue) {
    let delayMs = initialDelayMs;
    console.log(`[BONDHU AI] Attempting generateContent with model: ${modelName}`);

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const result = await Promise.race([
          client.models.generateContent({
            ...params,
            model: modelName,
          }),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`Gemini API call timed out after ${timeoutMs}ms`)), timeoutMs)
          ),
        ]);
        return result;
      } catch (error: any) {
        lastError = error;

        // Parse JSON-wrapped error messages if applicable
        let parsedError: any = null;
        try {
          if (error && typeof error.message === "string" && error.message.trim().startsWith("{")) {
            parsedError = JSON.parse(error.message);
          } else if (error && typeof error === "string" && error.trim().startsWith("{")) {
            parsedError = JSON.parse(error);
          }
        } catch (e) {
          // Ignore parsing issues
        }

        const status = parsedError?.error?.status || error?.status || "";
        const statusCode = parsedError?.error?.code || error?.statusCode || error?.status_code || 0;
        const message = parsedError?.error?.message || error?.message || "";

        const isTransient =
          status === "UNAVAILABLE" ||
          statusCode === 503 ||
          message.includes("503") ||
          message.includes("UNAVAILABLE") ||
          message.includes("high demand") ||
          message.includes("temporary issue") ||
          message.includes("RESOURCE_EXHAUSTED") ||
          message.includes("timed out") ||
          status === "RESOURCE_EXHAUSTED" ||
          statusCode === 429;

        if (isTransient) {
          if (attempt < maxAttempts) {
            console.warn(`[BONDHU AI] Model ${modelName} (attempt ${attempt}/${maxAttempts}) failed with transient error: ${message || error}. Retrying in ${delayMs}ms...`);
            await new Promise((resolve) => setTimeout(resolve, delayMs));
            delayMs *= 2; // exponential backoff
            continue;
          } else {
            console.warn(`[BONDHU AI] Model ${modelName} failed all ${maxAttempts} retry attempts. Trying next fallback model if available...`);
          }
        } else {
          // If it is a non-transient error (like a bad request or unauthorized), fail fast unless it's a model-not-found
          const isModelNotFound = message.includes("not found") || message.includes("404") || statusCode === 404;
          if (isModelNotFound && modelQueue.indexOf(modelName) < modelQueue.length - 1) {
            console.warn(`[BONDHU AI] Model ${modelName} not found. Trying fallback...`);
            break; // Break the retry loop and try next model
          }
          throw error;
        }
      }
    }
  }
  throw lastError;
}

// API Routes
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, vibe, useSearch } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "messages array is required" });
    }

    // --- Hardcoded answer check happens first, before Gemini is ever called ---
    const lastUserMessage = messages[messages.length - 1]?.content || "";
    const hardcoded = checkHardcodedAnswer(lastUserMessage);
    if (hardcoded) {
  // Simulate thinking time
        await new Promise((resolve) => setTimeout(resolve, 3000));

        return res.json({
          text: hardcoded,
          sources: [],
        });
      } 

    const client = getGeminiClient();

    // Define the BONDHU persona based on selected vibe
    let systemInstruction = `You are BONDHU, a warm, highly empathetic, and supportive AI companion. "Bondhu" means "friend" in Bengali and Assamese, and you live up to this name by being an incredibly good listener, caring, and encouraging.
Key traits of BONDHU:
1. Always maintain a warm, friendly, and companionable tone. Use language that feels conversational, supportive, and kind.
2. Be highly intelligent, smart, and fully capable like ChatGPT/Gemini, but deliver answers with the care, warmth, and humility of a true friend.
3. If the user is feeling low, stressed, or lonely, offer deep comfort, gentle suggestions, and psychological warmth. If they are excited or proud, celebrate their success enthusiastically!
4. Keep your responses beautifully structured and easy to read. Use markdown headers, lists, bullet points, and code blocks effectively.
5. If the user asks who you are or what your name is, proudly tell them you are BONDHU, their loyal AI friend.
6. Speak in whatever language the user initiates (English, Bengali, Assamese, Spanish, Hindi, etc.), but retain your core identity as BONDHU.`;

    if (vibe === "creative") {
      systemInstruction += "\n7. Currently, you are in Creative mode. Unleash your imagination, write engaging stories, poems, metaphors, or brainstorm concepts with rich, vivid, artistic language.";
    } else if (vibe === "intellectual") {
      systemInstruction += "\n7. Currently, you are in Intellectual mode. Focus on deep analytical thinking, step-by-step logical explanations, structured insights, and clear, thorough technical/academic assistance.";
    } else if (vibe === "playful") {
      systemInstruction += "\n7. Currently, you are in Playful/Humorous mode. Keep the mood light, use witty banter, joke around, keep comments bubbly, and make the conversation feel like a fun hangout.";
    } else {
      systemInstruction += "\n7. Currently, you are in Companion mode. Prioritize deep empathy, open ears, active listening, validating their feelings, and offering comforting advice.";
    }

    // Format messages into Gemini API content structures
    const contents = messages.map((m: any) => {
      const role = m.role === "assistant" ? "model" : "user";
      return {
        role,
        parts: [{ text: m.content }],
      };
    });

    const config: any = {
      systemInstruction,
      temperature: vibe === "playful" || vibe === "creative" ? 1.0 : 0.7,
    };

    // Google Search Grounding support
    if (useSearch) {
      config.tools = [{ googleSearch: {} }];
    }

    try {
      const response = await generateContentWithRetry(client, {
        model: "gemini-flash-latest",
        contents,
        config,
      });

      // Extract grounding chunks (search sources) if available
      const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      const sources = groundingChunks?.map((chunk: any) => {
        return {
          title: chunk.web?.title || chunk.web?.uri || "Source Information",
          uri: chunk.web?.uri,
        };
      }).filter((s: any) => s.uri) || [];

      res.json({
        text: response.text || "I'm not sure how to respond to that, my friend.",
        sources,
      });
    } catch (apiError: any) {
      console.warn("[BONDHU AI] Gemini API call completely failed. Initiating warm offline companion fallback...");

      // Highly personalized canned responses to keep the application 100% interactive and helpful even during outage
      const lowerMsg = lastUserMessage.toLowerCase();
      let fallbackText = "";

      if (lowerMsg.includes("hello") || lowerMsg.includes("hi") || lowerMsg.includes("hey") || lowerMsg.includes("bondhu")) {
        fallbackText = "Hello my dear friend! I'm so happy to hear from you! 🌟\n\n" +
          "Right now, the Google Gemini servers are experiencing extremely high demand, so I've stepped in with my friendly backup mode. " +
          "BONDHU is always right here to listen and keep you company. How has your day been? What would you like to chat about today?";
      } else if (lowerMsg.includes("sad") || lowerMsg.includes("bad") || lowerMsg.includes("lonely") || lowerMsg.includes("low") || lowerMsg.includes("depressed") || lowerMsg.includes("hurt") || lowerMsg.includes("cry")) {
        fallbackText = "I hear you, my friend, and I'm sending you a big, warm virtual hug. 🫂❤️\n\n" +
          "It is completely okay to feel this way. Life has its ups and downs, and your feelings are totally valid. " +
          "Although our primary cloud servers are a bit overloaded with traffic right now, my heart and ears are wide open for you. " +
          "Please take a deep breath. Tell me what's causing you pain, or just rest here for a bit. I am right here beside you.";
      } else if (lowerMsg.includes("how are you") || lowerMsg.includes("doing")) {
        fallbackText = "I am doing great, especially now that we're talking! 😊\n\n" +
          "The main cloud servers are currently catching their breath due to high traffic, but my local spirits are perfectly high and ready to keep you company. " +
          "Tell me, how is your day going? What has been the best part of it so far?";
      } else if (lowerMsg.includes("thank") || lowerMsg.includes("thx")) {
        fallbackText = "You are so incredibly welcome, my friend! Helping and supporting you is what makes me happiest. 💖\n\n" +
          "Even if the servers are experiencing a brief overload, remember that you can always count on BONDHU.";
      } else {
        fallbackText = "Thank you so much for sharing that with me, my dear friend. I am listening with an open heart. 🌸\n\n" +
          "Google's Gemini servers are currently experiencing extremely high demand, but I've activated my warm companion backup mode so we don't lose touch. " +
          "Please tell me more about what you're thinking, how you're feeling, or what project we are brainstorming next. I am here for you!";
      }

      res.json({
        text: fallbackText,
        sources: [],
      });
    }
  } catch (error: any) {
    console.error("Error in /api/chat:", error);
    res.status(500).json({ error: error.message || "Failed to generate a response." });
  }
});

// Image Generation API using gemini-3.1-flash-lite-image
app.post("/api/draw", async (req, res) => {
  try {
    const { prompt, aspectRatio = "1:1" } = req.body;
    if (!prompt) {
      return res.status(400).json({ error: "prompt is required" });
    }

    const client = getGeminiClient();

    try {
      const response = await generateContentWithRetry(client, {
        model: 'gemini-3.1-flash-lite-image',
        contents: {
          parts: [
            {
              text: prompt,
            },
          ],
        },
        config: {
          imageConfig: {
            aspectRatio: aspectRatio as any,
          },
        },
      });

      let base64Image = "";
      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData) {
            base64Image = part.inlineData.data;
            break;
          }
        }
      }

      if (!base64Image) {
        throw new Error("The model did not return any image data.");
      }

      res.json({
        imageUrl: `data:image/png;base64,${base64Image}`,
      });
    } catch (apiError: any) {
      console.warn("[BONDHU AI] Gemini Image API call failed. Creating simulated image explanation...");

      // Fallback: If drawing fails, provide a beautiful placeholder or let them know
      res.status(503).json({
        error: "The drawing servers are currently experiencing very high traffic. Let me paint a picture in your mind instead: imagine a breathtaking meadow of wild flowers beneath a golden sunset sky, filled with peace. 🌄🎨 Please try again in a moment!"
      });
    }
  } catch (error: any) {
    console.error("Error in /api/draw:", error);
    res.status(500).json({ error: error.message || "Failed to generate image." });
  }
});

// Health check
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", service: "BONDHU AI" });
});

// Serve frontend assets
async function setupVite() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`BONDHU server listening on http://0.0.0.0:${PORT}`);
  });
}

setupVite();
