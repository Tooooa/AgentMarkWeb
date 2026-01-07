import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, Box, Cpu, Plus, CheckCircle2, ChevronDown, Search } from 'lucide-react';
import { useI18n } from '../../i18n/I18nContext';
import { scenarios } from '../../data/mockData';

interface WelcomeScreenProps {
    onStart: (config: { scenarioId: string; payload: string; erasureRate: number }) => void;
    initialScenarioId: string;
    initialErasureRate: number;
}

type Mode = 'tool' | 'self' | 'add' | null;

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStart, initialScenarioId, initialErasureRate }) => {
    const { locale } = useI18n();
    const [selectedMode, setSelectedMode] = useState<Mode>(null);

    // Config State
    const [selectedScenarioId, setSelectedScenarioId] = useState<string>(''); // Start empty
    const [promptText, setPromptText] = useState('');
    const [showScenarioList, setShowScenarioList] = useState(false);

    const [payload, setPayload] = useState('AgentMark');
    const [erasureRate, setErasureRate] = useState(initialErasureRate);

    const modes = [
        { id: 'tool', title: 'Tool use', icon: Box, desc: 'Agent uses external tools' },
        { id: 'self', title: 'Self model', icon: Cpu, desc: 'Internal reasoning model' },
        { id: 'add', title: 'Add your agent', icon: Plus, desc: 'Custom agent integration' },
    ];

    const handleSelectScenario = (s: typeof scenarios[0]) => {
        setSelectedScenarioId(s.id);
        const title = locale === 'zh' ? s.title.zh : s.title.en;
        // Use the scenario's user query as the prompt text, or fallback to title
        setPromptText(s.userQuery || title);
        setShowScenarioList(false);
    };

    return (
        <div className="min-h-screen bg-white flex flex-col items-center justify-start pt-20 pb-10 px-4 font-sans">
            {/* Header / Branding */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-16"
            >
                <h1 className="text-4xl font-bold text-slate-900 mb-3">AgentMark</h1>
                <p className="text-slate-500 text-lg">Robust Watermarking for LLM Agents</p>
            </motion.div>

            {/* Step 1: Mode Selection Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-4xl mb-12">
                {modes.map((mode) => (
                    <motion.div
                        key={mode.id}
                        whileHover={{ y: -5, boxShadow: "0 20px 25px -5px rgb(0 0 0 / 0.1)" }}
                        whileTap={{ scale: 0.98 }}
                        onClick={() => setSelectedMode(mode.id as Mode)}
                        className={`cursor-pointer rounded-2xl p-8 border-2 transition-all flex flex-col items-center text-center gap-4 bg-white
                            ${selectedMode === mode.id
                                ? 'border-indigo-600 shadow-xl shadow-indigo-100 ring-4 ring-indigo-50'
                                : 'border-slate-100 shadow-sm hover:border-indigo-200'}`}
                    >
                        <div className={`p-4 rounded-full ${selectedMode === mode.id ? 'bg-indigo-600 text-white' : 'bg-slate-50 text-slate-600'}`}>
                            <mode.icon size={32} />
                        </div>
                        <h3 className={`text-xl font-bold ${selectedMode === mode.id ? 'text-indigo-900' : 'text-slate-700'}`}>
                            {mode.title}
                        </h3>
                        <p className="text-slate-400 text-sm">{mode.desc}</p>

                        {selectedMode === mode.id && (
                            <motion.div layoutId="check" className="absolute top-4 right-4 text-indigo-600">
                                <CheckCircle2 size={24} fill="currentColor" className="text-white" />
                            </motion.div>
                        )}
                    </motion.div>
                ))}
            </div>

            {/* Step 2: Configuration Panel (Expands when mode selected) */}
            <AnimatePresence>
                {selectedMode && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="w-full max-w-2xl overflow-visible"
                    >
                        <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-lg mb-10 relative z-0">
                            <h2 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
                                <span className="w-8 h-8 rounded-lg bg-indigo-100 text-indigo-600 flex items-center justify-center text-sm">02</span>
                                Configuration
                            </h2>

                            <div className="space-y-6">
                                {/* Prompt / Scenario Selector Input */}
                                <div className="relative z-50">
                                    <label className="block text-sm font-semibold text-slate-600 mb-2">Prompt</label>
                                    <div className="relative group">
                                        <div className="absolute left-4 top-3.5 text-slate-400 group-focus-within:text-indigo-500 transition-colors">
                                            <Search size={18} />
                                        </div>
                                        <input
                                            type="text"
                                            value={promptText}
                                            onChange={(e) => setPromptText(e.target.value)}
                                            onFocus={() => setShowScenarioList(true)}
                                            placeholder={locale === 'zh' ? "输入您的请求或选择场景..." : "Enter your query or select a scenario..."}
                                            className="w-full pl-11 pr-10 py-3 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 outline-none transition-all shadow-sm"
                                        />
                                        <button
                                            onClick={() => setShowScenarioList(!showScenarioList)}
                                            className="absolute right-3 top-3.5 text-slate-400 hover:text-indigo-600 transition-colors bg-transparent border-none cursor-pointer"
                                        >
                                            <ChevronDown size={18} />
                                        </button>
                                    </div>

                                    {/* Scenario Dropdown */}
                                    <AnimatePresence>
                                        {showScenarioList && (
                                            <motion.div
                                                initial={{ opacity: 0, y: 10, scale: 0.98 }}
                                                animate={{ opacity: 1, y: 0, scale: 1 }}
                                                exit={{ opacity: 0, y: 10, scale: 0.98 }}
                                                className="absolute top-full left-0 right-0 mt-2 bg-white rounded-xl border border-slate-200 shadow-xl overflow-hidden z-50 origin-top"
                                            >
                                                <div className="px-4 py-2 bg-slate-50 border-b border-slate-100 text-xs font-bold text-slate-500 uppercase tracking-wider flex justify-between items-center">
                                                    <span>{locale === 'zh' ? '预设场景' : 'Preset Scenarios'}</span>
                                                    <span className="text-[10px] bg-slate-200 px-1.5 py-0.5 rounded text-slate-500">Suggested</span>
                                                </div>
                                                <div className="max-h-60 overflow-y-auto p-1">
                                                    {scenarios.map(s => (
                                                        <button
                                                            key={s.id}
                                                            onClick={() => handleSelectScenario(s)}
                                                            className="w-full text-left px-4 py-3 hover:bg-indigo-50 rounded-lg text-sm text-slate-700 transition-colors flex items-center justify-between group"
                                                        >
                                                            <div className="flex flex-col gap-0.5">
                                                                <span className="font-medium group-hover:text-indigo-700">{locale === 'zh' ? s.title.zh : s.title.en}</span>
                                                                <span className="text-[10px] text-slate-400 truncate w-60">{s.userQuery}</span>
                                                            </div>
                                                            {selectedScenarioId === s.id && (
                                                                <motion.div layoutId="active-scenario">
                                                                    <CheckCircle2 size={16} className="text-indigo-600" />
                                                                </motion.div>
                                                            )}
                                                        </button>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>

                                {/* Overlay to close dropdown when clicking outside */}
                                {showScenarioList && (
                                    <div className="fixed inset-0 z-40" onClick={() => setShowScenarioList(false)}></div>
                                )}

                                {/* Watermark Payload & settings */}
                                <div className="grid grid-cols-2 gap-6 relative z-0">
                                    <div>
                                        <label className="block text-sm font-semibold text-slate-600 mb-2">Payload Content</label>
                                        <input
                                            type="text"
                                            value={payload}
                                            onChange={(e) => setPayload(e.target.value)}
                                            className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:border-indigo-500 outline-none transition-colors"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-semibold text-slate-600 mb-2">Loss Rate: {erasureRate}%</label>
                                        <input
                                            type="range"
                                            min="0" max="50" step="5"
                                            value={erasureRate}
                                            onChange={(e) => setErasureRate(Number(e.target.value))}
                                            className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600 mt-2"
                                        />
                                    </div>
                                </div>
                            </div>

                            <div className="mt-8 pt-6 border-t border-slate-100 flex justify-end relative z-0">
                                <button
                                    onClick={() => onStart({ scenarioId: selectedScenarioId || scenarios[0].id, payload, erasureRate })}
                                    className={`px-8 py-4 rounded-xl font-bold shadow-lg transition-all transform flex items-center gap-2
                                        ${selectedScenarioId || promptText
                                            ? 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:-translate-y-1'
                                            : 'bg-slate-200 text-slate-400 cursor-not-allowed'}`}
                                    disabled={!selectedScenarioId && !promptText}
                                >
                                    Proceed to Dashboard
                                    <ArrowRight size={20} />
                                </button>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default WelcomeScreen;
