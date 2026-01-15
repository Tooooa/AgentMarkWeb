import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Activity, Award, Columns, PlusCircle, Save, Search, Trash2 } from 'lucide-react';
import {
    CartesianGrid,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';
import MainLayout from '../layout/MainLayout';
import ComparisonView from '../layout/ComparisonView';
import FlowFeed from '../execution/FlowFeed';
import EvaluationModal from '../execution/EvaluationModal';
import ConfirmDialog from '../modals/ConfirmDialog';
import DecoderPanel from '../decoder/DecoderPanel';
import type { Step, Trajectory } from '../../data/mockData';
import { api } from '../../services/api';
import { useI18n } from '../../i18n/I18nContext';

type AddAgentDashboardProps = {
    onHome: () => void;
    apiKey: string;
    repoUrl: string;
    payload: string;
    erasureRate: number;
    setErasureRate: (val: number) => void;
    initialInput?: string;
};

const PROMPT_INSTRUCTION = `You are an action-selection assistant.
Return ONLY JSON with your probability over all candidate actions.
Example:
{
  "action_weights": {"Action1": 0.5, "Action2": 0.3, "Action3": 0.2},
  "action_args": {"Action1": {"arg": "value"}, "Action2": {"arg": "value"}, "Action3": {"arg": "value"}},
  "thought": "your thought here"
}
Requirements:
- action_weights MUST include every candidate (or top-K if instructed).
- All action_weights MUST be > 0. Use small values like 1e-3 for unlikely actions.
- action_args should include the chosen action. Other candidates may be omitted or left as empty objects.
  If you include other candidates, keep arguments minimal and consistent with the tool schema.
- Sum does not need to be exact; we will normalize.
- All action_weights must be > 0; do NOT return all zeros.
- Avoid uniform weights; if uncertain, break ties with slight preferences by candidate order.
- If a tool named "agentmark_score_actions" is available, call it with the JSON instead of writing text.
- Do NOT output any extra text or code fences.
- Provide a concise rationale in the thought field (no extra sections).`;

const buildGatewayPromptText = (trace: any, fallbackText: string) => {
    const messages = trace?.scoring_messages || trace?.execution_messages;
    if (Array.isArray(messages) && messages.length > 0) {
        const filtered = messages.filter((msg) => {
            const content = typeof msg?.content === 'string' ? msg.content : '';
            return content && !content.includes(PROMPT_INSTRUCTION) && !content.includes('[AgentMark mode=tools]');
        });
        if (filtered.length > 0) {
            return filtered.map((msg) => `${msg.role}: ${msg.content}`).join('\n');
        }
    }
    if (!fallbackText) return '';
    return fallbackText.replace(PROMPT_INSTRUCTION, '').replace('[AgentMark mode=tools]', '').trim();
};

const AddAgentDashboard: React.FC<AddAgentDashboardProps> = ({
    onHome,
    apiKey,
    repoUrl,
    payload,
    erasureRate,
    setErasureRate,
    initialInput
}) => {
    const { locale } = useI18n();
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [steps, setSteps] = useState<Step[]>([]);
    const [promptTraceText, setPromptTraceText] = useState('');
    const [baselinePromptTraceText, setBaselinePromptTraceText] = useState('');
    const [promptTrace, setPromptTrace] = useState<any>(null);
    const [baselinePromptTrace, setBaselinePromptTrace] = useState<any>(null);
    const [promptUserInput, setPromptUserInput] = useState('');
    const [historyScenarios, setHistoryScenarios] = useState<Trajectory[]>([]);
    const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);
    const [isSending, setIsSending] = useState(false);
    const [isComparisonMode, setIsComparisonMode] = useState(false);
    const [isEvaluating, setIsEvaluating] = useState(false);
    const [evaluationResult, setEvaluationResult] = useState<{
        model_a_score: number;
        model_b_score: number;
        reason: string;
    } | null>(null);
    const [isEvaluationModalOpen, setIsEvaluationModalOpen] = useState(false);
    const [isHistoryViewOpen, setIsHistoryViewOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [isBatchMode, setIsBatchMode] = useState(false);
    const [selectedScenarios, setSelectedScenarios] = useState<Set<string>>(new Set());
    const [isClearHistoryDialogOpen, setIsClearHistoryDialogOpen] = useState(false);
    const [isBatchDeleteDialogOpen, setIsBatchDeleteDialogOpen] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const erasedIndices = useMemo(() => new Set<number>(), []);
    const promptInputRef = useRef<HTMLInputElement>(null);
    const userStepIndexRef = useRef(-1);
    const revealSeqRef = useRef(0);
    const revealTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        return () => {
            if (revealTimeoutRef.current) clearTimeout(revealTimeoutRef.current);
        };
    }, []);
    const chartData = useMemo(() => {
        const data = steps.map((step, idx) => ({
            step: idx,
            tokens: step.metrics?.tokens ?? null,
            latency: step.metrics?.latency ?? null,
            baseTokens: step.baseline?.metrics?.tokens ?? null,
            baseLatency: step.baseline?.metrics?.latency ?? null
        }));
        if (data.length === 0) {
            data.push({
                step: 0,
                tokens: null,
                latency: null,
                baseTokens: null,
                baseLatency: null
            });
        }
        return data;
    }, [steps]);

    const gatewayPromptText = useMemo(() => {
        if (!promptTrace && !promptTraceText) return '';
        return buildGatewayPromptText(promptTrace, promptTraceText);
    }, [promptTrace, promptTraceText]);

    const formatPromptTrace = useCallback((trace: any) => {
        if (!trace) return '';
        const messages = trace.scoring_messages || trace.execution_messages;
        if (Array.isArray(messages) && messages.length > 0) {
            try {
                return JSON.stringify(messages, null, 2);
            } catch (err) {
                console.error('Failed to format prompt trace', err);
            }
        }
        return trace.scoring_prompt_text || trace.execution_prompt_text || '';
    }, []);

    const startSession = useCallback(async () => {
        const res = await api.addAgentStart(apiKey, repoUrl);
        setSessionId(res.sessionId);
        return res.sessionId;
    }, [apiKey, repoUrl]);

    const handleContinue = useCallback(async (prompt: string) => {
        if (isSending) return;
        const content = prompt.trim();
        if (!content) return;
        setIsSending(true);
        setPromptUserInput(content);
        if (evaluationResult) {
            setEvaluationResult(null);
            setIsEvaluationModalOpen(false);
        }
        if (baselinePromptTraceText) {
            setBaselinePromptTraceText('');
        }

        const userStepIndex = userStepIndexRef.current;
        userStepIndexRef.current -= 1;
        const placeholderStepIndex = userStepIndexRef.current;
        userStepIndexRef.current -= 1;

        const userInputStep: Step = {
            stepIndex: userStepIndex,
            thought: content,
            action: 'user_input',
            toolDetails: '',
            distribution: [],
            watermark: { bits: '', matrixRows: [], rankContribution: 0 },
            stepType: 'user_input',
            isHidden: false
        };

        const placeholderStep: Step = {
            stepIndex: placeholderStepIndex,
            thought: locale === 'zh' ? '思考中...' : 'Thinking...',
            action: '',
            toolDetails: '',
            distribution: [],
            watermark: { bits: '', matrixRows: [], rankContribution: 0 },
            stepType: 'other',
            isHidden: false
        };

        setSteps((prev) => [...prev, userInputStep, placeholderStep]);

        try {
            let sid = sessionId || await startSession();

            revealSeqRef.current += 1;
            const streamSeq = revealSeqRef.current;

            const applyPlaceholderPatch = (patch: Partial<Step>) => {
                setSteps((prev) => prev.map((s) => {
                    if (s.stepIndex !== placeholderStepIndex) return s;
                    return { ...s, ...patch };
                }));
            };

            try {
                await api.addAgentTurnStream(sid, content, apiKey, (evt) => {
                    if (revealSeqRef.current !== streamSeq) return;
                    if (!evt || typeof evt !== 'object') return;

                    if (evt.type === 'status') {
                        const state = evt.data?.state;
                        if (state === 'proxy_unavailable') {
                            applyPlaceholderPatch({ thought: locale === 'zh' ? '代理不可用，切换到直连...' : 'Proxy unavailable, switching to direct...' });
                        }
                        if (state === 'proxy_fallback') {
                            applyPlaceholderPatch({ thought: locale === 'zh' ? '代理异常，切换到直连...' : 'Proxy unstable, switching to direct...' });
                        }
                        if (state === 'retrying') {
                            applyPlaceholderPatch({ thought: locale === 'zh' ? '服务繁忙，正在重试...' : 'Service busy, retrying...' });
                        }
                    }

                    if (evt.type === 'thought_delta') {
                        const text = evt.data?.text;
                        if (typeof text === 'string' && text.trim()) {
                            applyPlaceholderPatch({ thought: text });
                        }
                    }

                    if (evt.type === 'tool_call') {
                        const name = evt.data?.name;
                        const args = evt.data?.arguments;
                        if (typeof name === 'string' && name) {
                            applyPlaceholderPatch({
                                action: `Call: ${name}`,
                                toolDetails: typeof args === 'string' ? args : (args ? JSON.stringify(args) : '')
                            });
                        }
                    }

                    if (evt.type === 'error') {
                        applyPlaceholderPatch({ thought: evt.message || (locale === 'zh' ? '出错了' : 'Error') });
                    }

                    if (evt.type === 'result') {
                        const res = evt.data;
                        const nextSteps = Array.isArray(res?.steps) ? (res.steps as Step[]) : [];

                        setSteps((prev) => {
                            const withoutPlaceholder = prev.filter((s) => s.stepIndex !== placeholderStepIndex);
                            return [...withoutPlaceholder, ...nextSteps];
                        });

                        const promptTrace = res?.promptTrace;
                        if (promptTrace) {
                            setPromptTrace(promptTrace);
                            setPromptTraceText(formatPromptTrace(promptTrace));
                        } else {
                            setPromptTrace(null);
                        }
                        const baseTrace = res?.baselinePromptTrace;
                        if (baseTrace) {
                            setBaselinePromptTrace(baseTrace);
                            setBaselinePromptTraceText(formatPromptTrace(baseTrace));
                        } else {
                            setBaselinePromptTrace(null);
                            setBaselinePromptTraceText('');
                        }
                    }
                });
            } catch (err: any) {
                const status = err?.response?.status;
                const msg = String(err?.message || '');
                if (status === 404 || msg.startsWith('HTTP 404')) {
                    applyPlaceholderPatch({ thought: locale === 'zh' ? '会话已过期，请重试' : 'Session expired, please retry' });
                    throw err;
                }
                throw err;
            }
        } catch (e) {
            console.error(e);
            setSteps((prev) => prev.filter((s) => s.stepIndex !== placeholderStepIndex));
        } finally {
            setIsSending(false);
        }
    }, [apiKey, isSending, sessionId, startSession, formatPromptTrace, evaluationResult, baselinePromptTraceText, locale]);

    useEffect(() => {
        if (initialInput) {
            handleContinue(initialInput);
        }
    }, [initialInput, handleContinue]);

    useEffect(() => {
        const loadScenarios = async () => {
            try {
                // Filter for add_agent scenarios
                const saved = await api.listScenarios('add_agent');
                setHistoryScenarios(saved);
            } catch (e) {
                console.error('Failed to load scenarios', e);
            }
        };
        loadScenarios();
    }, []);

    const handleSave = async () => {
        if (!sessionId || steps.length === 0) return;

        try {
            // Construct saving data
            const scenarioData = {
                id: sessionId,
                title: { en: "Add Agent Session", zh: "插件会话" }, // Default title
                taskName: "Add Agent Task",
                userQuery: steps[0]?.thought || "New Conversation",
                promptTrace: promptTraceText,
                baselinePromptTrace: baselinePromptTraceText,
                evaluation: evaluationResult || undefined,
                totalSteps: steps.length,
                steps: steps,
                type: 'add_agent'
            };

            // Generate better title if possible
            const firstMessage = steps[0]?.thought || "";
            const titlePreview = firstMessage.length > 30 ? firstMessage.substring(0, 30) + '...' : firstMessage;
            if (titlePreview) {
                scenarioData.title = { en: titlePreview, zh: titlePreview };
            }

            await api.saveScenario(scenarioData.title, scenarioData, sessionId, 'add_agent');

            // Refresh history
            const saved = await api.listScenarios('add_agent');
            setHistoryScenarios(saved);
            alert('Session saved!');
        } catch (e) {
            console.error("Failed to save session", e);
            alert("Failed to save session");
        }
    };

    const handleNewChat = () => {
        revealSeqRef.current += 1;
        if (revealTimeoutRef.current) {
            clearTimeout(revealTimeoutRef.current);
            revealTimeoutRef.current = null;
        }
        userStepIndexRef.current = -1;
        setSessionId(null);
        setSteps([]);
        setPromptTraceText('');
        setBaselinePromptTraceText('');
        setPromptTrace(null);
        setBaselinePromptTrace(null);
        setPromptUserInput('');
        setSelectedHistoryId(null);
        setEvaluationResult(null);
        setIsEvaluationModalOpen(false);
    };

    const handleEvaluate = useCallback(async () => {
        if (!sessionId || steps.length === 0) {
            alert(locale === 'zh' ? '没有可评估的对话' : 'No conversation to evaluate');
            return;
        }
        if (evaluationResult) {
            setIsEvaluationModalOpen(true);
            return;
        }
        setIsEvaluating(true);
        setIsEvaluationModalOpen(true);
        try {
            const result = await api.addAgentEvaluate(sessionId, locale);
            setEvaluationResult(result);
        } catch (e: any) {
            console.error('Add agent evaluation failed', e);
            const errorMsg = e.response?.data?.detail || e.message || 'Unknown error';
            alert(
                (locale === 'zh'
                    ? `评估失败: ${errorMsg}`
                    : `Evaluation failed: ${errorMsg}`) +
                '\n\n' +
                (locale === 'zh'
                    ? '请确认后端运行正常且已有对话步骤。'
                    : 'Please ensure the backend is running and there are steps to evaluate.')
            );
            setIsEvaluationModalOpen(false);
        } finally {
            setIsEvaluating(false);
        }
    }, [sessionId, steps.length, evaluationResult, locale]);

    const handleHistorySelect = (s: Trajectory) => {
        revealSeqRef.current += 1;
        if (revealTimeoutRef.current) {
            clearTimeout(revealTimeoutRef.current);
            revealTimeoutRef.current = null;
        }
        setSelectedHistoryId(s.id);
        setSessionId(s.id);
        const stepsWithBaseline = s.steps || [];
        setSteps(stepsWithBaseline);

        // Restore user query if available
        if (s.userQuery) {
            setPromptUserInput(s.userQuery);
        }

        setPromptTraceText(s.promptTrace || '');
        setBaselinePromptTraceText(s.baselinePromptTrace || '');

        // Auto-switch to comparison mode if baseline data exists
        const hasBaseline = stepsWithBaseline.some(step => step.baseline);
        setIsComparisonMode(hasBaseline);

        // Restore evaluation if available
        if (s.evaluation) {
            setEvaluationResult(s.evaluation);
        } else {
            setEvaluationResult(null);
        }

        // Reset modal state
        setIsEvaluationModalOpen(false);
        setIsHistoryViewOpen(false);
    };

    // Filter scenarios based on search query
    const filteredScenarios = useMemo(() => {
        if (!searchQuery.trim()) return historyScenarios;
        const query = searchQuery.toLowerCase();
        return historyScenarios.filter(s => {
            const titleEn = s.title?.en?.toLowerCase() || '';
            const titleZh = s.title?.zh?.toLowerCase() || '';
            const userQuery = s.userQuery?.toLowerCase() || '';
            return titleEn.includes(query) || titleZh.includes(query) || userQuery.includes(query);
        });
    }, [historyScenarios, searchQuery]);

    // Delete single scenario
    const handleDeleteScenario = async (id: string) => {
        try {
            await api.deleteScenario(id);
            const saved = await api.listScenarios('add_agent');
            setHistoryScenarios(saved);
            if (selectedHistoryId === id) {
                handleNewChat();
            }
        } catch (e) {
            console.error('Failed to delete scenario', e);
            alert(locale === 'zh' ? '删除失败' : 'Delete failed');
        }
    };

    // Batch delete scenarios
    const handleBatchDelete = async () => {
        if (selectedScenarios.size === 0) return;
        setIsDeleting(true);
        try {
            await api.batchDeleteScenarios(Array.from(selectedScenarios));
            const saved = await api.listScenarios('add_agent');
            setHistoryScenarios(saved);
            setSelectedScenarios(new Set());
            setIsBatchMode(false);
            if (selectedScenarios.has(selectedHistoryId || '')) {
                handleNewChat();
            }
        } catch (e) {
            console.error('Failed to batch delete', e);
            alert(locale === 'zh' ? '批量删除失败' : 'Batch delete failed');
        } finally {
            setIsDeleting(false);
            setIsBatchDeleteDialogOpen(false);
        }
    };

    // Clear all history (only add_agent type)
    const handleClearAllHistory = async () => {
        setIsDeleting(true);
        try {
            await api.clearHistoryByType('add_agent');
            setHistoryScenarios([]);
            handleNewChat();
        } catch (e) {
            console.error('Failed to clear history', e);
            alert(locale === 'zh' ? '清空失败' : 'Clear failed');
        } finally {
            setIsDeleting(false);
            setIsClearHistoryDialogOpen(false);
            setIsHistoryViewOpen(false);
        }
    };
    const leftPanel = (
        <div className="flex flex-col gap-6 h-full text-slate-900">
            <div className="rounded-2xl bg-white/85 border border-indigo-100/80 shadow-sm p-1 flex items-center gap-1">
                <button
                    onClick={() => setIsComparisonMode(false)}
                    className={`flex-1 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wider transition-all ${!isComparisonMode
                        ? 'bg-gradient-to-r from-sky-500 via-blue-600 to-indigo-600 text-white shadow shadow-blue-200/60'
                        : 'text-slate-500 hover:bg-sky-50'
                        }`}
                >
                    {locale === 'zh' ? '标准' : 'Standard'}
                </button>
                <button
                    onClick={() => setIsComparisonMode(true)}
                    className={`flex-1 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-1.5 ${isComparisonMode
                        ? 'bg-gradient-to-r from-teal-500 via-emerald-500 to-cyan-600 text-white shadow shadow-teal-200/60'
                        : 'text-slate-500 hover:bg-teal-50'
                        }`}
                >
                    <Columns size={12} />
                    {locale === 'zh' ? '对比' : 'Compare'}
                </button>
            </div>

            {!isComparisonMode && (
                <>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleNewChat}
                            className="flex-1 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-200"
                        >
                            <PlusCircle size={14} />
                            {locale === 'zh' ? '新对话' : 'New Chat'}
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={!sessionId || steps.length === 0}
                            className={`px-4 py-2.5 rounded-xl border border-indigo-200 text-xs font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${!sessionId || steps.length === 0
                                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                                : 'bg-white hover:bg-indigo-50 text-indigo-700 shadow-sm'
                                }`}
                        >
                            <Save size={14} />
                            {locale === 'zh' ? '保存' : 'Save'}
                        </button>
                    </div>

                    {/* History List */}
                    <div className="flex-1 bg-white/80 backdrop-blur-sm rounded-2xl border border-white/50 shadow-sm flex flex-col overflow-hidden">
                        <div className="p-4 border-b border-indigo-50/50 flex justify-between items-center">
                            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                                {locale === 'zh' ? '插件历史' : 'Plugin History'}
                            </h3>
                            {historyScenarios.length > 0 && (
                                <button
                                    onClick={() => setIsHistoryViewOpen(true)}
                                    className="text-[10px] text-indigo-500 hover:text-indigo-700 font-medium"
                                >
                                    {locale === 'zh' ? '查看全部' : 'View All'}
                                </button>
                            )}
                        </div>
                        <div className="flex-1 overflow-y-auto p-2 scrollbar-thin scrollbar-thumb-indigo-100/50 hover:scrollbar-thumb-indigo-200/50">
                            {historyScenarios.length === 0 ? (
                                <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2 opacity-60">
                                    <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
                                        <Search size={20} className="text-slate-300" />
                                    </div>
                                    <span className="text-xs">
                                        {locale === 'zh' ? '暂无记录' : 'No history yet'}
                                    </span>
                                </div>
                            ) : (
                                historyScenarios.slice(0, 5).map((s) => (
                                    <div
                                        key={s.id}
                                        className={`p-3 rounded-xl mb-2 cursor-pointer transition-all border group relative ${selectedHistoryId === s.id
                                            ? 'bg-gradient-to-r from-indigo-50 to-blue-50 border-indigo-200 shadow-sm'
                                            : 'hover:bg-white border-transparent hover:border-indigo-50 hover:shadow-sm'
                                            }`}
                                    >
                                        <div onClick={() => handleHistorySelect(s)} className="flex-1">
                                            <div className="text-xs font-bold text-slate-700 line-clamp-1 mb-1 pr-6">
                                                {locale === 'zh' ? (s.title.zh || s.title.en) : s.title.en}
                                            </div>
                                            <div className="flex items-center justify-between text-[10px] text-slate-400">
                                                <span>{s.steps.length} turns</span>
                                                <span>{new Date(s.createdAt || Date.now()).toLocaleDateString()}</span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                if (confirm(locale === 'zh' ? '确定要删除这条记录吗？' : 'Delete this conversation?')) {
                                                    handleDeleteScenario(s.id);
                                                }
                                            }}
                                            className="absolute top-2 right-2 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-50 rounded transition-all"
                                            title={locale === 'zh' ? '删除' : 'Delete'}
                                        >
                                            <Trash2 size={12} className="text-slate-400 hover:text-red-500" />
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </>
            )}
            {
                isComparisonMode && (
                    <>
                        <button
                            onClick={handleEvaluate}
                            disabled={isEvaluating}
                            className={`w-full py-2.5 rounded-xl text-[11px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow-sm ${isEvaluating
                                ? 'bg-teal-50 text-teal-500 cursor-wait'
                                : 'bg-gradient-to-r from-teal-400 via-emerald-500 to-cyan-500 text-white shadow-teal-200/60 hover:shadow-emerald-200/60'
                                }`}
                        >
                            {isEvaluating ? (
                                <>
                                    <div className="w-3 h-3 border-2 border-teal-200 border-t-teal-500 rounded-full animate-spin"></div>
                                    {locale === 'zh' ? '评估中...' : 'Evaluating...'}
                                </>
                            ) : (
                                <>
                                    <Award size={14} className="text-teal-100" />
                                    {locale === 'zh' ? '评估智能体' : 'Evaluate Agents'}
                                </>
                            )}
                        </button>
                        {evaluationResult && (
                            <div className="bg-white/90 rounded-xl border border-teal-100/80 p-3 shadow-sm flex items-center justify-between">
                                <div className="flex flex-col items-center flex-1 border-r border-teal-100/60">
                                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                                        {locale === 'zh' ? '基础' : 'Base'}
                                    </span>
                                    <div className="flex items-center text-sm font-bold text-slate-700">
                                        {Number(evaluationResult.model_a_score).toFixed(1)}
                                        <span className="text-[10px] text-slate-400 ml-0.5">/10</span>
                                    </div>
                                </div>
                                <div className="flex flex-col items-center flex-1">
                                    <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-wider">
                                        {locale === 'zh' ? '水印' : 'Ours'}
                                    </span>
                                    <div className="flex items-center text-sm font-bold text-emerald-600">
                                        {Number(evaluationResult.model_b_score).toFixed(1)}
                                        <span className="text-[10px] text-emerald-300 ml-0.5">/10</span>
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                )
            }



            <div className="flex-1 bg-white/80 rounded-2xl shadow-[0_20px_45px_-30px_rgba(15,23,42,0.4)] border border-teal-100/70 ring-1 ring-teal-100/60 p-5 flex flex-col min-h-0">
                <div className="flex items-center gap-2 text-teal-800 border-b border-teal-100/70 pb-2 mb-3 shrink-0">
                    <Activity size={16} className="text-teal-500" />
                    <h3 className="font-bold text-xs uppercase tracking-wide">Utility Monitor</h3>
                </div>

                <div className="flex-1 flex flex-col gap-2 min-h-0 overflow-y-auto pr-1">
                    <div className="flex-1 min-h-[100px] flex flex-col">
                        <div className="flex justify-between items-center mb-1 shrink-0">
                            <span className="text-[10px] font-semibold text-slate-600">Token Throughput</span>
                            <div className="flex gap-2 text-[8px]">
                                <span className="flex items-center gap-0.5">
                                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-600"></span> Ours
                                </span>
                                <span className="flex items-center gap-0.5 text-slate-400">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span> Base
                                </span>
                            </div>
                        </div>
                        <div className="flex-1 w-full min-h-0 overflow-hidden relative">
                            <ResponsiveContainer width="99%" height="100%" debounce={50}>
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e7e5e4" />
                                    <XAxis dataKey="step" hide />
                                    <YAxis tick={{ fontSize: 9 }} axisLine={false} tickLine={false} width={20} />
                                    <Tooltip
                                        contentStyle={{ fontSize: '10px' }}
                                        itemStyle={{ padding: 0 }}
                                        wrapperStyle={{ zIndex: 1000 }}
                                        isAnimationActive={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="baseTokens"
                                        stroke="#94a3b8"
                                        strokeWidth={2}
                                        strokeDasharray="4 4"
                                        dot={false}
                                        isAnimationActive={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="tokens"
                                        stroke="#0f766e"
                                        strokeWidth={2}
                                        dot={{ r: 2 }}
                                        activeDot={{ r: 4 }}
                                        isAnimationActive={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="flex-1 min-h-[100px] flex flex-col">
                        <div className="flex justify-between items-center mb-1 shrink-0">
                            <span className="text-[10px] font-semibold text-slate-600">Step Latency (s)</span>
                            <div className="flex gap-2 text-[8px]">
                                <span className="flex items-center gap-0.5">
                                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-500"></span> Ours
                                </span>
                                <span className="flex items-center gap-0.5 text-slate-400">
                                    <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span> Base
                                </span>
                            </div>
                        </div>
                        <div className="flex-1 w-full min-h-0 overflow-hidden relative">
                            <ResponsiveContainer width="99%" height="100%" debounce={50}>
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e7e5e4" />
                                    <XAxis dataKey="step" hide />
                                    <YAxis tick={{ fontSize: 9 }} axisLine={false} tickLine={false} width={20} />
                                    <Tooltip
                                        contentStyle={{ fontSize: '10px' }}
                                        itemStyle={{ padding: 0 }}
                                        wrapperStyle={{ zIndex: 1000 }}
                                        isAnimationActive={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="baseLatency"
                                        stroke="#94a3b8"
                                        strokeWidth={2}
                                        strokeDasharray="4 4"
                                        dot={false}
                                        isAnimationActive={false}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="latency"
                                        stroke="#f97316"
                                        strokeWidth={2}
                                        dot={{ r: 2 }}
                                        activeDot={{ r: 4 }}
                                        isAnimationActive={false}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>
        </div >
    );

    return (
        <div className="relative h-screen overflow-hidden bg-slate-50">
            {/* Ambient Background - Blue/Indigo Theme for Standard Mode */}
            <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
                <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full bg-indigo-200/30 blur-3xl opacity-60 mix-blend-multiply" />
                <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] rounded-full bg-blue-100/40 blur-3xl opacity-60 mix-blend-multiply" />
                <div className="absolute top-[20%] left-[10%] w-[400px] h-[400px] rounded-full bg-sky-100/30 blur-3xl opacity-40 mix-blend-multiply" />
            </div>

            <MainLayout
                variant="add_agent"
                layout={isComparisonMode ? 'compare' : 'standard'}
                left={leftPanel}
                middle={
                    isComparisonMode ? (
                        <ComparisonView
                            visibleSteps={steps}
                            erasedIndices={erasedIndices}
                            scenarioId={sessionId || undefined}
                            userQueryLeft={gatewayPromptText || promptUserInput}
                            userQueryRight={promptUserInput ? `${promptUserInput}\n\n${PROMPT_INSTRUCTION}` : PROMPT_INSTRUCTION}
                            promptInstruction={PROMPT_INSTRUCTION}
                            evaluationResult={evaluationResult}
                            variant="add_agent"
                        />
                    ) : (
                        <FlowFeed
                            visibleSteps={steps}
                            erasedIndices={erasedIndices}
                            userQuery={promptTraceText}
                            userQueryLabel="LLM Prompt"
                            userQueryFormat="json"
                            userInputHighlight={promptUserInput}
                            userInputLabel="User Input"
                            onContinue={handleContinue}
                            isPlaying={isSending}
                            promptInputRef={promptInputRef}
                            variant="add_agent"
                        />
                    )
                }
                right={
                    isComparisonMode ? null : (
                        <DecoderPanel
                            visibleSteps={steps}
                            erasedIndices={erasedIndices}
                            targetPayload={payload}
                            erasureRate={erasureRate}
                            setErasureRate={setErasureRate}
                            promptInputRef={promptInputRef}
                            variant="add_agent"
                        />
                    )
                }
                onHome={onHome}
            />
            <EvaluationModal
                isOpen={isEvaluationModalOpen}
                onClose={() => setIsEvaluationModalOpen(false)}
                result={evaluationResult}
                isLoading={isEvaluating}
                variant="add_agent"
            />

            {/* Full Screen History View Modal */}
            {isHistoryViewOpen && (
                <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-xl shadow-2xl w-full max-w-6xl h-[80vh] flex flex-col">
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b">
                            <h2 className="text-2xl font-bold text-gray-800">
                                {locale === 'zh' ? '插件历史记录' : 'Plugin History'}
                                {isBatchMode && selectedScenarios.size > 0 && (
                                    <span className="ml-3 text-sm font-normal text-blue-600">
                                        ({selectedScenarios.size} {locale === 'zh' ? '已选择' : 'selected'})
                                    </span>
                                )}
                            </h2>
                            <div className="flex items-center gap-2">
                                {/* Batch Mode Toggle */}
                                {filteredScenarios.length > 0 && !isBatchMode && (
                                    <button
                                        onClick={() => {
                                            setIsBatchMode(true);
                                            setSelectedScenarios(new Set());
                                        }}
                                        className="px-3 py-1.5 text-xs text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-md transition-colors flex items-center gap-1.5 border border-blue-200"
                                        title={locale === 'zh' ? '批量删除' : 'Batch Delete'}
                                    >
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
                                        </svg>
                                        <span>{locale === 'zh' ? '批量删除' : 'Batch Delete'}</span>
                                    </button>
                                )}

                                {/* Batch Mode Actions */}
                                {isBatchMode && (
                                    <>
                                        <button
                                            onClick={() => {
                                                if (selectedScenarios.size === filteredScenarios.length) {
                                                    setSelectedScenarios(new Set());
                                                } else {
                                                    setSelectedScenarios(new Set(filteredScenarios.map(s => s.id)));
                                                }
                                            }}
                                            className="px-3 py-1.5 text-xs text-gray-700 bg-white hover:bg-gray-50 rounded-md transition-colors flex items-center gap-1.5 border border-gray-200"
                                        >
                                            <span>{selectedScenarios.size === filteredScenarios.length
                                                ? (locale === 'zh' ? '取消全选' : 'Deselect All')
                                                : (locale === 'zh' ? '全选' : 'Select All')
                                            }</span>
                                        </button>

                                        <button
                                            onClick={() => {
                                                if (selectedScenarios.size > 0) {
                                                    setIsBatchDeleteDialogOpen(true);
                                                }
                                            }}
                                            disabled={selectedScenarios.size === 0}
                                            className={`px-3 py-1.5 text-xs rounded-md transition-colors flex items-center gap-1.5 border ${selectedScenarios.size > 0
                                                ? 'text-red-700 bg-red-50 hover:bg-red-100 border-red-200'
                                                : 'text-gray-400 bg-gray-50 border-gray-200 cursor-not-allowed'
                                                }`}
                                        >
                                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                            <span>{locale === 'zh' ? '删除选中' : 'Delete Selected'}</span>
                                        </button>

                                        <button
                                            onClick={() => {
                                                setIsBatchMode(false);
                                                setSelectedScenarios(new Set());
                                            }}
                                            className="px-3 py-1.5 text-xs text-gray-700 bg-white hover:bg-gray-50 rounded-md transition-colors border border-gray-200"
                                        >
                                            {locale === 'zh' ? '取消' : 'Cancel'}
                                        </button>
                                    </>
                                )}

                                {/* Clear All Button */}
                                {filteredScenarios.length > 0 && !isBatchMode && (
                                    <button
                                        onClick={() => {
                                            setIsClearHistoryDialogOpen(true);
                                        }}
                                        className="px-3 py-1.5 text-xs text-gray-700 bg-white hover:bg-gray-50 rounded-md transition-colors flex items-center gap-1.5 border border-gray-200"
                                        title={locale === 'zh' ? '清空所有历史记录' : 'Clear All History'}
                                    >
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                        <span>{locale === 'zh' ? '清空历史' : 'Clear'}</span>
                                    </button>
                                )}

                                <button
                                    onClick={() => {
                                        setIsHistoryViewOpen(false);
                                        setIsBatchMode(false);
                                        setSelectedScenarios(new Set());
                                    }}
                                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                                    title={locale === 'zh' ? '关闭' : 'Close'}
                                >
                                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            </div>
                        </div>

                        {/* Search Bar */}
                        <div className="px-6 pt-4 pb-2">
                            <div className="relative">
                                <input
                                    type="text"
                                    placeholder={locale === 'zh' ? '搜索对话...' : 'Search conversations...'}
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full px-4 py-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                                <svg
                                    className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                                </svg>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6">
                            {filteredScenarios.length === 0 ? (
                                <div className="flex flex-col items-center justify-center h-full text-gray-400">
                                    <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                                    </svg>
                                    <p className="text-lg font-medium">{locale === 'zh' ? '暂无历史记录' : 'No history'}</p>
                                </div>
                            ) : (
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {filteredScenarios.map((scenario) => {
                                        const isSelected = selectedScenarios.has(scenario.id);
                                        return (
                                            <div
                                                key={scenario.id}
                                                className={`p-4 rounded-lg border-2 transition-all hover:shadow-lg relative ${isBatchMode
                                                    ? isSelected
                                                        ? 'border-blue-500 bg-blue-50 cursor-pointer'
                                                        : 'border-gray-200 hover:border-blue-300 cursor-pointer'
                                                    : selectedHistoryId === scenario.id
                                                        ? 'border-blue-500 bg-blue-50 cursor-pointer'
                                                        : 'border-gray-200 hover:border-blue-300 cursor-pointer'
                                                    }`}
                                                onClick={() => {
                                                    if (isBatchMode) {
                                                        const newSelected = new Set(selectedScenarios);
                                                        if (isSelected) {
                                                            newSelected.delete(scenario.id);
                                                        } else {
                                                            newSelected.add(scenario.id);
                                                        }
                                                        setSelectedScenarios(newSelected);
                                                    } else {
                                                        handleHistorySelect(scenario);
                                                    }
                                                }}
                                            >
                                                {/* Batch Mode Checkbox */}
                                                {isBatchMode && (
                                                    <div className="absolute top-2 left-2 z-10">
                                                        <div className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${isSelected
                                                            ? 'bg-blue-500 border-blue-500'
                                                            : 'bg-white border-gray-300'
                                                            }`}>
                                                            {isSelected && (
                                                                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                                                                </svg>
                                                            )}
                                                        </div>
                                                    </div>
                                                )}

                                                <div className="flex items-start justify-between mb-2">
                                                    <h3 className={`font-semibold text-gray-800 flex-1 line-clamp-2 ${isBatchMode ? 'ml-7' : ''}`}>
                                                        {locale === 'zh' ? (scenario.title.zh || scenario.title.en) : scenario.title.en}
                                                    </h3>
                                                    {!isBatchMode && (
                                                        <div className="flex items-center gap-1 ml-2">
                                                            {/* Delete Button */}
                                                            <button
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    if (window.confirm(locale === 'zh' ? '确定要删除这条记录吗？' : 'Delete this conversation?')) {
                                                                        handleDeleteScenario(scenario.id);
                                                                    }
                                                                }}
                                                                className="p-1 hover:bg-red-50 rounded transition-colors"
                                                                title={locale === 'zh' ? '删除' : 'Delete'}
                                                            >
                                                                <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                                </svg>
                                                            </button>
                                                        </div>
                                                    )}
                                                </div>
                                                <p className="text-sm text-gray-500 line-clamp-2 mb-2">
                                                    {scenario.userQuery || (locale === 'zh' ? '暂无内容' : 'No content')}
                                                </p>
                                                <div className="flex items-center justify-between text-xs text-gray-400">
                                                    <span>{scenario.steps?.length || scenario.totalSteps || 0} {locale === 'zh' ? '步' : 'steps'}</span>
                                                    {!isBatchMode && selectedHistoryId === scenario.id && (
                                                        <span className="text-blue-500 font-medium">
                                                            {locale === 'zh' ? '当前' : 'Active'}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* Clear History Confirmation Dialog */}
            <ConfirmDialog
                isOpen={isClearHistoryDialogOpen}
                onClose={() => !isDeleting && setIsClearHistoryDialogOpen(false)}
                onConfirm={handleClearAllHistory}
                title={locale === 'zh' ? '清空插件历史记录' : 'Clear Plugin History'}
                message={locale === 'zh'
                    ? '此操作将永久删除所有插件历史会话记录，包括：\n\n• 所有对话内容\n• 所有水印数据\n• 所有评估结果\n\n此操作不可撤销，是否确认清空？'
                    : 'This action will permanently delete all plugin conversation history, including:\n\n• All conversation content\n• All watermark data\n• All evaluation results\n\nThis action cannot be undone. Are you sure you want to clear all history?'}
                confirmText={isDeleting ? (locale === 'zh' ? '删除中...' : 'Deleting...') : (locale === 'zh' ? '确认清空' : 'Clear All')}
                cancelText={locale === 'zh' ? '取消' : 'Cancel'}
                isDestructive={true}
            />

            {/* Batch Delete Confirmation Dialog */}
            <ConfirmDialog
                isOpen={isBatchDeleteDialogOpen}
                onClose={() => !isDeleting && setIsBatchDeleteDialogOpen(false)}
                onConfirm={handleBatchDelete}
                title={locale === 'zh' ? '批量删除' : 'Batch Delete'}
                message={locale === 'zh'
                    ? `确定要删除选中的 ${selectedScenarios.size} 条记录吗？\n\n此操作不可撤销。`
                    : `Are you sure you want to delete ${selectedScenarios.size} selected conversation(s)?\n\nThis action cannot be undone.`}
                confirmText={isDeleting ? (locale === 'zh' ? '删除中...' : 'Deleting...') : (locale === 'zh' ? '确认删除' : 'Delete')}
                cancelText={locale === 'zh' ? '取消' : 'Cancel'}
                isDestructive={true}
            />
        </div>
    );
};

export default AddAgentDashboard;
