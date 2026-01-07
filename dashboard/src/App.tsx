import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import MainLayout from './components/layout/MainLayout';
import ControlPanel from './components/controls/ControlPanel';
import FlowFeed from './components/execution/FlowFeed';
import DecoderPanel from './components/decoder/DecoderPanel';
import WelcomeScreen from './components/layout/WelcomeScreen';
import { useSimulation } from './hooks/useSimulation';
import { I18nProvider } from './i18n/I18nContext';

function App() {
  const {
    scenarios,
    activeScenario,
    activeScenarioId,
    setActiveScenarioId,
    isPlaying,
    setIsPlaying,
    currentStepIndex,
    erasureRate,
    setErasureRate,
    erasedIndices,
    handleReset,
    handleNext,
    handlePrev,
    visibleSteps,
    isLiveMode,
    setIsLiveMode,
    apiKey,
    setApiKey,
    handleInitSession
  } = useSimulation();

  const [hasStarted, setHasStarted] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [targetPayload, setTargetPayload] = useState('AgentMark'); // Used for display

  const handleStart = (config: { scenarioId: string; payload: string; erasureRate: number }) => {
    setActiveScenarioId(config.scenarioId);
    setTargetPayload(config.payload);
    setErasureRate(config.erasureRate);
    setHasStarted(true);
  };

  return (
    <I18nProvider>
      <div className="bg-slate-50 min-h-screen">
        <AnimatePresence mode='wait'>
          {!hasStarted ? (
            <WelcomeScreen
              key="welcome"
              onStart={handleStart}
              initialScenarioId={activeScenarioId}
              initialErasureRate={erasureRate}
            />
          ) : (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              className="h-screen overflow-hidden"
            >
              <MainLayout
                left={
                  <ControlPanel
                    scenarios={scenarios}
                    activeScenarioId={activeScenarioId}
                    onSelectScenario={setActiveScenarioId}
                    isPlaying={isPlaying}
                    onTogglePlay={() => setIsPlaying(!isPlaying)}
                    onReset={handleReset}
                    onNext={handleNext}
                    onPrev={handlePrev}
                    erasureRate={erasureRate}
                    setErasureRate={setErasureRate}
                    currentStep={currentStepIndex}
                    totalSteps={activeScenario.totalSteps}
                    isLiveMode={isLiveMode}
                    onToggleLiveMode={() => setIsLiveMode(!isLiveMode)}
                    apiKey={apiKey}
                    setApiKey={setApiKey}
                    onInitSession={handleInitSession}
                  />
                }
                middle={
                  <FlowFeed
                    visibleSteps={visibleSteps}
                    erasedIndices={erasedIndices}
                  />
                }
                right={
                  <DecoderPanel
                    visibleSteps={visibleSteps}
                    erasedIndices={erasedIndices}
                    targetPayload={targetPayload} // Pass payload for display
                  />
                }
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </I18nProvider>
  );
}

export default App;
