import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import MainLayout from './components/layout/MainLayout';
import ControlPanel from './components/controls/ControlPanel';
import FlowFeed from './components/execution/FlowFeed';
import DecoderPanel from './components/decoder/DecoderPanel';
import ComparisonView from './components/layout/ComparisonView';
import WelcomeScreen from './components/layout/WelcomeScreen';
import SaveScenarioModal from './components/modals/SaveScenarioModal';
import SettingsModal from './components/modals/SettingsModal';
import EvaluationModal from './components/execution/EvaluationModal';
import TutorialTooltip from './components/tutorial/TutorialTooltip';
import { useSimulation } from './hooks/useSimulation';
import { I18nProvider } from './i18n/I18nContext';

function App() {
  const {
    scenarios,
    activeScenario,
    activeScenarioId,
    setActiveScenarioId,
    refreshScenarios, // New
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
    handleInitSession,
    setCustomQuery,
    setApiKey,
    customQuery,
    sessionId,
    apiKey, // Add back
    payload, // from hook
    setPayload, // from hook
    handleContinue, // from hook
    handleNewConversation: startNewConversation,

    // Evaluation
    evaluateSession,
    isEvaluating,
    evaluationResult,
    isEvaluationModalOpen,
    setIsEvaluationModalOpen,

    // Comparison
    isComparisonMode,
    setIsComparisonMode
  } = useSimulation();

  const [hasStarted, setHasStarted] = useState(false);
  // const [isComparisonMode, setIsComparisonMode] = useState(false); // Removed
  const [isSaveModalOpen, setIsSaveModalOpen] = useState(false);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false);
  const [isFirstEntry, setIsFirstEntry] = useState(true);
  
  // 新手引导状态 - 每次刷新页面后都会出现一次
  const [hasCompletedTutorial, setHasCompletedTutorial] = useState(false);
  const [showTutorial, setShowTutorial] = useState(false);
  const [tutorialStep, setTutorialStep] = useState(1);
  
  // 引导目标元素的ref
  const settingsButtonRef = useRef<HTMLButtonElement>(null);
  const channelNoiseRef = useRef<HTMLDivElement>(null);
  const modeToggleRef = useRef<HTMLDivElement>(null);
  const promptInputRef = useRef<HTMLInputElement>(null);


  // NOTE: targetPayload is now redundant if we use hook's payload?
  // But DecoderPanel uses it. Let's keep using hook's payload for consistency
  // or sync them.
  // Actually, let's just use the hook's payload directly for DecoderPanel too, 
  // but DecoderPanel prop is 'targetPayload'.

  const handleStart = (config: { scenarioId: string; payload: string; erasureRate: number; query?: string }) => {
    setActiveScenarioId(config.scenarioId);
    if (config.query) {
      setCustomQuery(config.query);
    } else {
      setCustomQuery(""); // Reset if not custom
    }
    setPayload(config.payload); // Sync to hook
    setErasureRate(config.erasureRate);
    // Note: setHasStarted logic will trigger the effect below
    setHasStarted(true);
  };

  const handleNewConversation = useCallback(async () => {
    // 1. Auto-Save & Reset via Hook
    await startNewConversation();
    // 2. Stay on Dashboard (do not reset hasStarted)
    // setHasStarted(false); // REMOVED
  }, [startNewConversation]);

  // 处理新手引导下一步
  const handleTutorialNext = () => {
    if (tutorialStep < 4) {
      setTutorialStep(tutorialStep + 1);
    } else {
      // 教程完成，在当前会话中标记完成（刷新页面后会重置）
      setShowTutorial(false);
      setTutorialStep(1);
      setHasCompletedTutorial(true);
    }
  };

  // 获取当前步骤的目标ref
  const getCurrentTutorialRef = () => {
    switch (tutorialStep) {
      case 1: return settingsButtonRef;
      case 2: return channelNoiseRef;
      case 3: return modeToggleRef;
      case 4: return promptInputRef;
      default: return null;
    }
  };

  // 初次进入主页面时自动弹出设置窗口
  useEffect(() => {
    if (hasStarted && isFirstEntry) {
      setIsSettingsModalOpen(true);
      setIsFirstEntry(false);
    }
  }, [hasStarted, isFirstEntry]);

  // 当设置窗口关闭时，如果是首次进入且未完成教程，则启动新手引导
  useEffect(() => {
    if (!isSettingsModalOpen && !isFirstEntry && !hasCompletedTutorial && !showTutorial && tutorialStep === 1) {
      // 等待设置窗口关闭动画完成后再启动引导
      setTimeout(() => {
        setShowTutorial(true);
      }, 500);
    }
  }, [isSettingsModalOpen, isFirstEntry, hasCompletedTutorial, showTutorial, tutorialStep]);

  // 移除自动初始化逻辑，由用户在设置中明确点击“应用”来启动会话
  // useEffect(() => {
  //   if (hasStarted && isLiveMode && customQuery && !sessionId) {
  //     handleInitSession();
  //   }
  // }, [hasStarted, isLiveMode, customQuery, sessionId, handleInitSession]);

  const liveStats = useMemo(() => {
    if (activeScenario && activeScenario.steps.length > 0) {
      const metricsSteps = activeScenario.steps.filter(s => s.metrics);
      if (metricsSteps.length > 0) {
        const totalLat = metricsSteps.reduce((sum, s) => sum + (s.metrics?.latency || 0), 0);
        const totalTok = metricsSteps.reduce((sum, s) => sum + (s.metrics?.tokens || 0), 0);
        // Latency in ms (from seconds)
        // Tokens/sec = totalTokens / totalTime
        return {
          avgLatency: parseFloat((totalLat * 1000 / metricsSteps.length).toFixed(2)),
          avgTokens: totalLat > 0 ? parseFloat((totalTok / totalLat).toFixed(0)) : 0
        };
      }
    }
    return undefined;
  }, [activeScenario]);

  const commonControlPanel = (
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
        isComparisonMode={isComparisonMode}
        onToggleComparisonMode={() => setIsComparisonMode(!isComparisonMode)}
        liveStats={liveStats}
        currentScenario={activeScenario}
        onNew={handleNewConversation}
        onSave={() => setIsSaveModalOpen(true)}

        // Evaluation
        onEvaluate={evaluateSession}
        isEvaluating={isEvaluating}
        evaluationResult={evaluationResult}
        
        // Tutorial ref
        modeToggleRef={modeToggleRef}
      />
  );

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
              isLiveMode={isLiveMode}
              onToggleLiveMode={() => setIsLiveMode(!isLiveMode)}
              apiKey={apiKey}
              setApiKey={setApiKey}
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
              {isComparisonMode ? (
                // Comparison Mode Layout
                <div className="h-full p-4 grid grid-cols-[340px_1fr] gap-6 overflow-hidden max-w-[1920px] mx-auto">
                  <div className="h-full overflow-hidden">
                    {commonControlPanel}
                  </div>
                  <div className="h-full overflow-hidden">
                    <ComparisonView
                      visibleSteps={visibleSteps}
                      erasedIndices={erasedIndices}
                      scenarioId={activeScenario.id}
                      evaluationResult={evaluationResult}
                    />
                  </div>
                </div>
              ) : (
                // Standard Dashboard Layout
                <MainLayout
                  left={commonControlPanel}
                  middle={
                    <FlowFeed
                      visibleSteps={visibleSteps}
                      erasedIndices={erasedIndices}
                      userQuery={customQuery || activeScenario.userQuery}
                      onContinue={handleContinue}
                      isPlaying={isPlaying}
                      onTogglePlay={() => setIsPlaying(!isPlaying)}
                      scenarioId={activeScenario.id}
                      promptInputRef={promptInputRef}
                    />
                  }
                  right={
                    <DecoderPanel
                      visibleSteps={visibleSteps}
                      erasedIndices={erasedIndices}
                      targetPayload={payload}
                      erasureRate={erasureRate}
                      setErasureRate={setErasureRate}
                      channelNoiseRef={channelNoiseRef}
                    />
                  }
                  onHome={() => {
                    setHasStarted(false);
                    handleReset();
                  }}
                  onSettings={() => setIsSettingsModalOpen(true)}
                  settingsButtonRef={settingsButtonRef}
                />
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <SettingsModal
          isOpen={isSettingsModalOpen}
          onClose={() => setIsSettingsModalOpen(false)}
          isLiveMode={isLiveMode}
          onToggleLiveMode={() => setIsLiveMode(!isLiveMode)}
          apiKey={apiKey}
          setApiKey={setApiKey}
          customQuery={customQuery}
          setCustomQuery={setCustomQuery}
          payload={payload}
          setPayload={setPayload}
          onInitSession={handleInitSession}
          hasActiveConversation={!!sessionId || (activeScenario.steps && activeScenario.steps.length > 0)}
        />

        <SaveScenarioModal
          isOpen={isSaveModalOpen}
          onClose={() => setIsSaveModalOpen(false)}
          scenarioData={activeScenario}
          onSaved={() => {
            refreshScenarios();
            // Optional: maybe confirm to user
          }}
        />

        <EvaluationModal
          isOpen={isEvaluationModalOpen}
          onClose={() => setIsEvaluationModalOpen(false)}
          result={evaluationResult}
          isLoading={isEvaluating}
        />

        {/* 新手引导 */}
        <TutorialTooltip
          isOpen={showTutorial}
          step={tutorialStep}
          totalSteps={4}
          onNext={handleTutorialNext}
          targetRef={getCurrentTutorialRef()}
        />
      </div>
    </I18nProvider>
  );
}

export default App;
