import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.8.0';

const userInput = document.getElementById('userInput');
const messages = document.getElementById('messages');

let docs, embeddings;

async function loadData() {
  const [docRes, embRes] = await Promise.all([
    fetch("website_documents.json"),
    fetch("document_embeddings.json")
  ]);
  const docJson = await docRes.json();
  docs = Object.values(docJson);
  embeddings = await embRes.json();
}

// Load models from local folder
const embedder = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', {
  local_files_only: true,
  cache_dir: './models'
});
const generator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M', {
  local_files_only: true,
  cache_dir: './models'
});

const KEYWORD_RESPONSES = [
  // Existing 17
  { keywords: ["courses", "available", "offerings", "training"], response: "We offer courses in Microsoft Azure, Chip Design, 5G, Cybersecurity, Data Science, Generative AI, Full Stack Development, Quantum Computing, and HPC." },
  { keywords: ["founder", "ceo", "head"], response: "The founder and CEO of Object Automation is Ganesan Narayanasamy." },
  { keywords: ["location", "address", "where located"], response: "We are located in California, USA (9500 Gilman Drive, La Jolla) and in India. Contact us at +1 914 204 2581 or +91 7397784815." },
  { keywords: ["contact", "reach", "email", "phone number"], response: "You can contact us via our contact page: https://object-automation.com/html/home/contact.html" },
  { keywords: ["azure", "microsoft cloud"], response: "The Microsoft Azure course includes cloud fundamentals, virtual networks, DevOps pipelines, and more." },
  { keywords: ["chip design", "vlsi", "semiconductor"], response: "Our Chip Design course covers VLSI principles, ASIC flow, and semiconductor logic fundamentals." },
  { keywords: ["generative ai", "genai", "ai course"], response: "The Generative AI course teaches LLMs, prompt engineering, transformers, and project-based learning." },
  { keywords: ["quantum computing", "quantum", "qubit"], response: "The Quantum Computing course includes Qiskit programming, quantum logic gates, and entanglement simulations." },
  { keywords: ["full stack", "web development", "mern"], response: "Our Full Stack Development course covers React, Node.js, MongoDB, and REST APIs." },
  { keywords: ["data science", "machine learning", "ml"], response: "The Data Science course includes Python, Pandas, ML models, data visualization, and real-world projects." },
  { keywords: ["hpc", "high performance computing", "gpu"], response: "The HPC course includes parallel computing, GPU acceleration using CUDA, and cluster architecture." },
  { keywords: ["5g", "fifth generation"], response: "We explore 5G innovations across industries like smart factories, automotive, and healthcare." },
  { keywords: ["ai project", "health bot"], response: "Our Health Bot is an AI-powered healthcare assistant for quick triage and symptom check." },
  { keywords: ["events", "webinars", "seminars"], response: "We conduct regular online webinars and in-person events. Check our events page for updates." },
  { keywords: ["internship", "job", "career"], response: "To apply for internships or job opportunities, contact us through our website or submit your resume." },
  { keywords: ["certification", "certificate", "proof"], response: "Yes, we provide industry-recognized certificates after course completion." },
  { keywords: ["fee", "payment", "emi"], response: "We accept payments via UPI, cards, and offer EMI options on request." },
  { keywords: ["recording", "missed class"], response: "If you miss a class, recordings will be provided or you may join the next batch." },

  // ✅ 13 additional student-focused responses
  { keywords: ["eligibility", "who can apply"], response: "Our courses are open to students, professionals, and graduates from any background with an interest in tech." },
  { keywords: ["beginner", "no experience", "basic knowledge"], response: "Yes, beginners are welcome! We provide foundational training for students with no prior experience." },
  { keywords: ["duration", "how long", "course length"], response: "Each course typically lasts between 4 to 12 weeks depending on the topic." },
  { keywords: ["live class", "zoom", "instructor led"], response: "Yes, we conduct live instructor-led classes through platforms like Zoom." },
  { keywords: ["project", "hands-on", "practical"], response: "All our courses include hands-on projects to apply your knowledge in real-world scenarios." },
  { keywords: ["doubt", "mentor", "support"], response: "You’ll get access to doubt-clearing sessions and personal mentorship throughout your course." },
  { keywords: ["community", "peer", "group"], response: "We offer community learning spaces where you can collaborate with peers and join project groups." },
  { keywords: ["resume", "linkedin", "portfolio"], response: "We help you build your tech portfolio and improve your resume/LinkedIn with certifications and projects." },
  { keywords: ["exam", "test", "assessment"], response: "Courses include assessments or mini-projects to evaluate your understanding and provide feedback." },
  { keywords: ["class timing", "schedule", "batch"], response: "We have flexible batch timings, including evening and weekend options." },
  { keywords: ["language", "medium", "english"], response: "All our courses are conducted in English for global accessibility." },
  { keywords: ["certificate validity", "recognition"], response: "Our certificates are recognized by hiring managers and industry professionals globally." },
  { keywords: ["rejoin", "re-enroll", "repeat class"], response: "You can rejoin missed sessions or future batches with prior notice." }
];

// Match static response
function getKeywordResponse(query) {
  const lowerQuery = query.toLowerCase();
  for (const entry of KEYWORD_RESPONSES) {
    if (entry.keywords.some(kw => lowerQuery.includes(kw))) {
      return entry.response;
    }
  }
  return null;
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

function getRelevantDocs(queryEmbedding, k = 5) {
  const sims = embeddings.map((emb, i) => ({
    index: i,
    score: cosineSimilarity(queryEmbedding, emb)
  }));
  sims.sort((a, b) => b.score - a.score);
  return sims.slice(0, k).map(sim => docs[sim.index]);
}

function searchByKeyword(query) {
  const lower = query.toLowerCase();
  return docs.filter(d => d.text.toLowerCase().includes(lower));
}

function buildPrompt(context, query) {
  return `
Answer the question based **only** on the context below.
If the answer is not in the context, say "Sorry, I couldn't find that in the website content."

📄 Context:
${context}

❓ Question:
${query}

💬 Answer:`.trim();
}

userInput.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') {
    const query = userInput.value.trim();
    if (!query) return;

    userInput.value = '';
    messages.innerHTML += `<div class="user">🧍 ${query}</div>`;

    const staticAnswer = getKeywordResponse(query);
    if (staticAnswer) {
      messages.innerHTML += `<div class="bot">🤖 ${staticAnswer}</div>`;
      messages.scrollTop = messages.scrollHeight;
      return;
    }

    try {
      const qEmbedding = await embedder(query, { pooling: 'mean', normalize: true });
      const vectorDocs = getRelevantDocs(Array.from(qEmbedding.data));
      const keywordDocs = searchByKeyword(query);

      const combinedDocs = [...new Map([...keywordDocs, ...vectorDocs].map(d => [d.url, d])).values()];
      const topContext = combinedDocs.slice(0, 5).map(d =>
        `Source: ${d.url}\n${d.text.slice(0, 1000)}`
      ).join("\n\n").slice(0, 2000);

      if (!topContext.trim()) {
        messages.innerHTML += `<div class="bot">🤖 Sorry, I couldn't find that in the website content.</div>`;
        return;
      }

      const prompt = buildPrompt(topContext, query);
      const response = await generator(prompt, { max_new_tokens: 256 });

      let reply = "Sorry, no answer was generated.";
      if (Array.isArray(response) && response.length > 0 && response[0]?.generated_text) {
        reply = response[0].generated_text.trim();
      }

      messages.innerHTML += `<div class="bot">🤖 ${reply}</div>`;
      messages.scrollTop = messages.scrollHeight;

    } catch (err) {
      console.error("❌ Error during generation:", err);
      messages.innerHTML += `<div class="bot">❌ Error: ${err.message}</div>`;
    }
  }
});

await loadData();