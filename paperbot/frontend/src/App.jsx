import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Brain, FileImage, FileText, GitBranch } from 'lucide-react';
import ChatWindow from './components/ChatWindow.jsx';
import ImageModal from './components/ImageModal.jsx';

const features = [
  {
    icon: Brain,
    title: 'RAG Retrieval',
    copy: 'Semantic search over indexed paper chunks finds the passages that matter before the model answers.',
  },
  {
    icon: GitBranch,
    title: 'LangGraph Pipeline',
    copy: 'A graph workflow routes questions through retrieval, generation, and optional visual lookup.',
  },
  {
    icon: FileImage,
    title: 'Smart Suggestions',
    copy: 'PaperBot detects when a relevant figure or table exists and suggests it below the answer.',
  },
  {
    icon: FileText,
    title: 'Source Citations',
    copy: 'Every answer is grounded in specific chunks with source filename and page number cited inline.',
  },
];

const papers = [
  {
    num: '01',
    title: 'BERT',
    desc: 'Bidirectional transformer pretraining for language understanding.',
    tag: 'NLP · BERT',
  },
  {
    num: '02',
    title: 'RoBERTa',
    desc: 'A robustly optimized BERT training recipe for stronger language representations.',
    tag: 'NLP · BERT',
  },
  {
    num: '03',
    title: 'Sentiment Models',
    desc: 'Comparative sentiment analysis methods and evaluation findings.',
    tag: 'NLP · Sentiment',
  },
  {
    num: '04',
    title: 'Attention',
    desc: 'Transformer attention mechanisms, architecture notes, and interpretability clues.',
    tag: 'Transformers',
  },
  {
    num: '05',
    title: 'Evaluation',
    desc: 'Metrics, tables, confusion matrices, and paper-level evidence summaries.',
    tag: 'Metrics',
  },
];

function useThreeParticles() {
  useEffect(() => {
    const canvas = document.getElementById('particle-bg');
    const THREE = window.THREE;
    if (!canvas || !THREE) return undefined;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 9;

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);

    const particleCount = 600;
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 22;
      positions[i + 1] = (Math.random() - 0.5) * 14;
      positions[i + 2] = (Math.random() - 0.5) * 10;
    }

    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particleMaterial = new THREE.PointsMaterial({
      color: 0x4f8ef7,
      size: 0.022,
      transparent: true,
      opacity: 0.15,
    });
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener('resize', handleResize);

    let frameId = 0;
    const animate = () => {
      frameId = window.requestAnimationFrame(animate);
      particles.rotation.y += 0.0004;
      particles.rotation.x += 0.0001;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.cancelAnimationFrame(frameId);
      window.removeEventListener('resize', handleResize);
      particleGeometry.dispose();
      particleMaterial.dispose();
      renderer.dispose();
    };
  }, []);
}

export default function App() {
  const [sessionId, setSessionId] = useState('');
  const [health, setHealth] = useState('checking');
  const [modalImage, setModalImage] = useState(null);
  const askButtonRef = useRef(null);

  useThreeParticles();

  useEffect(() => {
    setSessionId(crypto.randomUUID());
    const checkHealth = async () => {
      try {
        const response = await axios.get('/api/health');
        setHealth(response.data?.status === 'ok' ? 'ok' : 'offline');
      } catch {
        setHealth('offline');
      }
    };
    checkHealth();
  }, []);

  const openChat = () => {
    askButtonRef.current?.click();
  };

  return (
    <>
      <div className="site-shell">
        <nav className="nav">
          <a href="#top" className="logo">
            🔬 PaperBot
          </a>
          <div className="nav-links">
            <a href="#papers">Papers</a>
            <a href="#features">Features</a>
            <a href="#about">About</a>
            <button type="button" className="cta" onClick={openChat}>
              Ask PaperBot
            </button>
          </div>
        </nav>

        <main id="top">
          <section className="hero">
            <div className="eyebrow">Research · AI · 5 Papers Indexed</div>
            <h1>Ask anything about the papers.</h1>
            <p className="hero-sub">
              Powered by LangGraph RAG with semantic retrieval, smart content suggestions,
              and cited answers from BERT, RoBERTa, and Sentiment Analysis research.
            </p>
            <div className="hero-buttons">
              <button type="button" className="btn-primary" onClick={openChat}>
                Start Asking →
              </button>
              <a href="#papers" className="ghost-btn">
                See the Papers ↓
              </a>
            </div>
            <div className="hero-stats">
              <span className="hero-stats-item">5 Papers</span>
              <div className="hero-stats-divider" />
              <span className="hero-stats-item">48 Pages</span>
              <div className="hero-stats-divider" />
              <span className="hero-stats-item">LangGraph</span>
            </div>
          </section>

          <section className="section" id="features">
            <div className="section-header">
              <div>
                <span className="label">How It Works</span>
                <h2>From question to cited answer.</h2>
              </div>
              <p className="section-copy">
                PaperBot treats every answer as a retrieval task, then routes figure requests
                through smart content suggestions before showing extracted page imagery.
              </p>
            </div>
            <div className="feature-grid">
              {features.map((feature) => {
                const Icon = feature.icon;
                return (
                  <article className="feature-card" key={feature.title}>
                    <div className="feature-icon">
                      <Icon size={22} />
                    </div>
                    <h3>{feature.title}</h3>
                    <p>{feature.copy}</p>
                  </article>
                );
              })}
            </div>
          </section>

          <section className="section" id="papers">
            <div className="section-header">
              <div>
                <span className="label">Indexed Papers</span>
                <h2>5 papers. Every chunk. Instantly searchable.</h2>
              </div>
              <p className="section-copy">
                Ask across transformer foundations, sentiment analysis evidence, model comparisons,
                and figure-backed results.
              </p>
            </div>
            <div className="paper-list">
              {papers.map((paper) => (
                <div className="paper-row" key={paper.num}>
                  <span className="paper-num">{paper.num}</span>
                  <div className="paper-info">
                    <div className="paper-title">{paper.title}</div>
                    <div className="paper-desc">{paper.desc}</div>
                  </div>
                  <span className="paper-tag">{paper.tag}</span>
                </div>
              ))}
            </div>
          </section>
        </main>

        <footer className="footer" id="about">
          <div className="logo">🔬 PaperBot</div>
          <div>Built with LangGraph + Qdrant + Groq</div>
        </footer>

        <div className="health-badge" data-status={health}>
          {health === 'ok'
            ? '● API online'
            : health === 'checking'
              ? '● checking...'
              : '● API offline'}
        </div>
      </div>

      <ChatWindow
        sessionId={sessionId}
        launcherRef={askButtonRef}
        onOpenImage={setModalImage}
      />
      <ImageModal image={modalImage} onClose={() => setModalImage(null)} />
    </>
  );
}
