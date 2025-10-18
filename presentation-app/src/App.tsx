import { useState, useEffect, useCallback } from 'react';
import { slides, getTotalDuration } from './slides';
import {
  ThroughputRaceChart,
  QuantumCountdown,
  PerformanceHeatmap,
  SecurityThermometer,
  DAGVisualization,
  DAG3DVisualization,
} from './components';
import './App.css';

function App() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(slides[0].duration);
  const [progress, setProgress] = useState(0);

  const totalSlides = slides.length;
  const totalDuration = getTotalDuration();

  const goToNextSlide = useCallback(() => {
    if (currentSlide < totalSlides - 1) {
      setCurrentSlide(currentSlide + 1);
      setTimeRemaining(slides[currentSlide + 1].duration);
    } else {
      setIsPlaying(false);
      setCurrentSlide(0);
      setTimeRemaining(slides[0].duration);
    }
  }, [currentSlide, totalSlides]);

  const goToPreviousSlide = () => {
    if (currentSlide > 0) {
      setCurrentSlide(currentSlide - 1);
      setTimeRemaining(slides[currentSlide - 1].duration);
    }
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const resetPresentation = () => {
    setCurrentSlide(0);
    setIsPlaying(false);
    setTimeRemaining(slides[0].duration);
  };

  useEffect(() => {
    let timer: number | undefined;

    if (isPlaying) {
      timer = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 0.1) {
            goToNextSlide();
            return slides[currentSlide + 1]?.duration || 0;
          }
          return prev - 0.1;
        });
      }, 100);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, currentSlide, goToNextSlide]);

  useEffect(() => {
    const completedTime = slides.slice(0, currentSlide).reduce((sum, s) => sum + s.duration, 0);
    const currentProgress = slides[currentSlide].duration - timeRemaining;
    setProgress(((completedTime + currentProgress) / totalDuration) * 100);
  }, [currentSlide, timeRemaining, totalDuration]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowRight':
        case ' ':
          if (!isPlaying) goToNextSlide();
          break;
        case 'ArrowLeft':
          if (!isPlaying) goToPreviousSlide();
          break;
        case 'p':
        case 'P':
          togglePlayPause();
          break;
        case 'r':
        case 'R':
          resetPresentation();
          break;
        case 'Escape':
          setIsPlaying(false);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentSlide, isPlaying, goToNextSlide]);

  const slide = slides[currentSlide];

  return (
    <div className="presentation">
      <div className="progress-bar" style={{ width: `${progress}%` }} />

      <div className="slide-container">
        <div className="slide-header">
          <div className="header-left">
            <img src="/logos/logo-2.png" alt="Quillon Logo" className="header-logo" />
            <h1 className="slide-title">{slide.title}</h1>
          </div>
          <div className="slide-meta">
            Slide {currentSlide + 1} / {totalSlides}
          </div>
        </div>

        <div className="slide-content">
          {slide.centerLogo && (
            <div className="center-logo-container">
              <img src={slide.centerLogo} alt="Quillon Logo" className="center-logo" />
            </div>
          )}

          {slide.chart && (
            <div className="chart-container">
              {slide.chart === 'throughput-race' && <ThroughputRaceChart />}
              {slide.chart === 'quantum-countdown' && <QuantumCountdown />}
              {slide.chart === 'performance-heatmap' && <PerformanceHeatmap />}
              {slide.chart === 'security-thermometer' && <SecurityThermometer />}
              {slide.chart === 'dag-visualization' && <DAGVisualization />}
              {slide.chart === 'dag-3d' && <DAG3DVisualization />}
            </div>
          )}

          {slide.code ? (
            <div className="code-section">
              {slide.content.length > 0 && (
                <div className="code-intro">
                  {slide.content.map((line, i) => (
                    <p key={i} className="content-line">{line}</p>
                  ))}
                </div>
              )}
              <pre className="code-block">
                <code className={`language-${slide.language || 'rust'}`}>
                  {slide.code}
                </code>
              </pre>
            </div>
          ) : !slide.chart ? (
            <div className="text-section">
              {slide.content.map((line, i) => {
                if (line === '') {
                  return <div key={i} className="spacer" />;
                }

                const isHeader = line.match(/^[🔗⚖️🔐🌐🚀🔮⚡⏱️📊🛡️📢✅⚓🔄🌐📡🔍🎲🔮🌈🎨📈🔍📈🎓🎬🧪💾💿♻️🔗📚💬🤝📄🎥🔧📊⚛️]/);
                const isBullet = line.startsWith('•') || line.startsWith('-');
                const isIndented = line.startsWith('   ');

                return (
                  <p
                    key={i}
                    className={`content-line ${isHeader ? 'header' : ''} ${isBullet ? 'bullet' : ''} ${isIndented ? 'indented' : ''}`}
                  >
                    {line}
                  </p>
                );
              })}
            </div>
          ) : null}

          {slide.explanation && (
            <div className="slide-explanation">
              <div className="explanation-icon">💡</div>
              <div className="explanation-text">{slide.explanation}</div>
            </div>
          )}
        </div>
      </div>

      <div className="controls">
        <button onClick={goToPreviousSlide} disabled={currentSlide === 0} className="control-btn">
          ◀ Previous
        </button>

        <button onClick={togglePlayPause} className="control-btn play-btn">
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>

        <button onClick={goToNextSlide} disabled={currentSlide === totalSlides - 1} className="control-btn">
          Next ▶
        </button>

        <button onClick={resetPresentation} className="control-btn">
          ↺ Reset
        </button>

        <div className="timer">
          {Math.ceil(timeRemaining)}s remaining
        </div>
      </div>

      <div className="keyboard-hints">
        <span>Space/→: Next</span>
        <span>←: Previous</span>
        <span>P: Play/Pause</span>
        <span>R: Reset</span>
        <span>Esc: Stop</span>
      </div>
    </div>
  );
}

export default App;
