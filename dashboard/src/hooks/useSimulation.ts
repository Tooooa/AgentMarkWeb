
import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { scenarios } from '../data/mockData';
import type { Trajectory, Step } from '../data/mockData';
import { api } from '../services/api';


export const useSimulation = () => {
    // Static State
    const [activeScenarioId, setActiveScenarioId] = useState<string>(scenarios[0].id);

    // Live State
    const [isLiveMode, setIsLiveMode] = useState(false);
    const [apiKey, setApiKey] = useState("sk-7f4e6c78384e4aaab0eb0c59af411618");
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [liveScenario, setLiveScenario] = useState<Trajectory | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    // Common State
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [erasureRate, setErasureRate] = useState(0);
    const [erasedIndices, setErasedIndices] = useState<Set<number>>(new Set());

    // Derived Active Scenario
    const activeScenario = useMemo(() => {
        if (isLiveMode && liveScenario) {
            return liveScenario;
        }
        return scenarios.find(s => s.id === activeScenarioId) || scenarios[0];
    }, [activeScenarioId, isLiveMode, liveScenario]);

    const timerRef = useRef<number | null>(null);

    const handleReset = useCallback(() => {
        setIsPlaying(false);
        setCurrentStepIndex(0);
        setErasedIndices(new Set());
        if (isLiveMode) {
            // In live mode, reset means clear steps? Or re-init? 
            // For now, let's keep it simple: stop whatever is happening
        }
    }, [isLiveMode]);

    // Live Session Init
    const handleInitSession = useCallback(async () => {
        if (!apiKey) return;
        setIsLoading(true);
        try {
            const data = await api.initSession(apiKey, activeScenarioId);
            setSessionId(data.sessionId);
            setLiveScenario({
                id: data.task.id,
                title: { en: "Live Session", zh: "实时会话" }, // Can be improved
                taskName: "Live Execution",
                userQuery: data.task.query,
                totalSteps: 0,
                steps: []
            });
            setCurrentStepIndex(0);
            setErasedIndices(new Set());
        } catch (e) {
            console.error(e);
            alert("Failed to init session");
        } finally {
            setIsLoading(false);
        }
    }, [apiKey, activeScenarioId]);

    const handleNext = useCallback(async () => {
        // Live Mode Logic
        if (isLiveMode) {
            if (!sessionId || !liveScenario) return;

            // Prevent stepping if we are already at the end of what we have fetched...
            // UNLESS we are fetching a NEW step.
            // Logic: "Next" in live mode triggers generation of the next step.

            setIsLoading(true);
            try {
                const stepData = await api.step(sessionId);

                // Convert API step to Dashboard Step
                if (stepData.done && stepData.action === "Finish" && !stepData.thought) {
                    // Just finish signal
                    return;
                }

                const newStep: Step = {
                    stepIndex: liveScenario.steps.length,
                    thought: stepData.thought,
                    action: stepData.action,
                    distribution: stepData.distribution || [],
                    watermark: {
                        bits: stepData.watermark?.bits || "",
                        matrixRows: stepData.watermark?.matrixRows || [],
                        rankContribution: stepData.watermark?.rankContribution || 0
                    },
                    stepType: stepData.done ? 'finish' : 'tool'
                };

                setLiveScenario(prev => {
                    if (!prev) return null;
                    return {
                        ...prev,
                        totalSteps: prev.steps.length + 1,
                        steps: [...prev.steps, newStep]
                    };
                });

                // Move cursor to showing this new step
                setCurrentStepIndex(prev => prev + 1);

            } catch (e) {
                console.error(e);
            } finally {
                setIsLoading(false);
            }
            return;
        }

        // Static Logic
        if (currentStepIndex < activeScenario.totalSteps) {
            setCurrentStepIndex(prev => prev + 1);
            if (Math.random() * 100 < erasureRate) {
                setErasedIndices(old => new Set(old).add(currentStepIndex));
            }
        }
    }, [currentStepIndex, activeScenario, erasureRate, isLiveMode, sessionId, liveScenario]);

    const handlePrev = useCallback(() => {
        if (currentStepIndex > 0) {
            setCurrentStepIndex(prev => prev - 1);
        }
    }, [currentStepIndex]);

    // Timer Logic (Auto Play)
    useEffect(() => {
        if (isPlaying) {
            timerRef.current = window.setInterval(async () => {
                if (isLiveMode) {
                    if (isLoading) return; // Wait for request
                    // Check if done logic needed? 
                    // For now, user stops manually or we detect end
                    await handleNext();
                } else {
                    setCurrentStepIndex(prev => {
                        if (prev >= activeScenario.totalSteps) {
                            setIsPlaying(false);
                            return prev;
                        }
                        const nextStepIdx = prev;
                        if (Math.random() * 100 < erasureRate) {
                            setErasedIndices(old => new Set(old).add(nextStepIdx));
                        }
                        return prev + 1;
                    });
                }
            }, isLiveMode ? 2000 : 1000); // Slower for live mode API calls
        } else {
            if (timerRef.current) clearInterval(timerRef.current);
        }
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [isPlaying, activeScenario, erasureRate, isLiveMode, isLoading, handleNext]);

    return {
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
        visibleSteps: activeScenario.steps.slice(0, currentStepIndex),
        // Live Mode Props
        isLiveMode,
        setIsLiveMode,
        apiKey,
        setApiKey,
        handleInitSession,
        isLoading
    };
};
