
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, MessageSquare } from 'lucide-react';
import { useI18n } from '../../i18n/I18nContext';
import { scenarios } from '../../data/mockData';

interface WelcomeScreenProps {
    onStart: (config: { scenarioId: string; payload: string; erasureRate: number }) => void;
    initialScenarioId: string;
    initialErasureRate: number;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStart, initialScenarioId, initialErasureRate }) => {
    const { locale } = useI18n();
    const [selectedScenarioId, setSelectedScenarioId] = useState(initialScenarioId);
    const [payload, setPayload] = useState('AgentMark');
    const [erasureRate, setErasureRate] = useState(initialErasureRate);

    return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-white rounded-3xl shadow-2xl border border-slate-100 max-w-2xl w-full overflow-hidden flex flex-col md:flex-row min-h-[500px]"
            >
                {/* Left Side: Visual/Branding */}
                <div className="w-full md:w-2/5 bg-gradient-to-br from-indigo-600 to-violet-700 p-8 flex flex-col justify-between text-white relative overflow-hidden">
                    <div className="z-10">
                        <div className="flex items-center gap-2 mb-6">
                            <div className="bg-white/20 p-2 rounded-lg backdrop-blur-sm">
                                <MessageSquare size={24} className="text-white" />
                            </div>
                            <span className="font-bold text-xl tracking-wide">AgentMark</span>
                        </div>
                        <h1 className="text-3xl font-bold mb-4 leading-tight">
                            {locale === 'zh' ? '开启您的水印之旅' : 'Start Your Journey'}
                        </h1>
                        <p className="text-indigo-100 text-sm leading-relaxed opacity-90">
                            {locale === 'zh'
                                ? '探索大模型代理的鲁棒性水印嵌入与验证过程。'
                                : 'Explore robust watermark embedding and verification for LLM Agents.'}
                        </p>
                    </div>

                    {/* Decorative Elements */}
                    <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-white/10 rounded-full blur-2xl"></div>
                    <div className="absolute top-10 -left-10 w-32 h-32 bg-indigo-500/20 rounded-full blur-xl"></div>
                </div>

                {/* Right Side: Controls */}
                <div className="w-full md:w-3/5 p-8 flex flex-col gap-6">
                    <div className="space-y-4">
                        {/* Scenario Selector */}
                        <div>
                            <label className="block text-sm font-semibold text-slate-700 mb-2">
                                {locale === 'zh' ? '选择演示场景' : 'Select Scenario'}
                            </label>
                            <div className="grid gap-2">
                                {scenarios.map(s => (
                                    <div
                                        key={s.id}
                                        onClick={() => setSelectedScenarioId(s.id)}
                                        className={`cursor-pointer p-3 rounded-xl border transition-all flex items-center justify-between ${selectedScenarioId === s.id
                                            ? 'bg-indigo-50 border-indigo-500 shadow-sm'
                                            : 'bg-white border-slate-200 hover:border-indigo-200 hover:bg-slate-50'
                                            }`}
                                    >
                                        <div>
                                            <p className={`text-sm font-medium ${selectedScenarioId === s.id ? 'text-indigo-900' : 'text-slate-700'}`}>
                                                {locale === 'zh' ? s.title.zh : s.title.en}
                                            </p>
                                            <p className="text-[10px] text-slate-400 mt-0.5">{s.totalSteps} steps</p>
                                        </div>
                                        {selectedScenarioId === s.id && (
                                            <div className="w-4 h-4 rounded-full bg-indigo-500 flex items-center justify-center">
                                                <div className="w-1.5 h-1.5 rounded-full bg-white"></div>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Payload Input */}
                        <div>
                            <label className="block text-sm font-semibold text-slate-700 mb-2">
                                {locale === 'zh' ? '输入水印载荷 (Payload)' : 'Watermark Payload'}
                            </label>
                            <input
                                type="text"
                                value={payload}
                                onChange={(e) => setPayload(e.target.value)}
                                className="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none text-sm transition-all bg-slate-50 focus:bg-white"
                                placeholder="Enter text or bits..."
                            />
                        </div>

                        {/* Erasure Rate Slider */}
                        <div>
                            <div className="flex justify-between items-center mb-2">
                                <label className="text-sm font-semibold text-slate-700">
                                    {locale === 'zh' ? '日志擦除率 (Simulated Loss)' : 'Log Erasure Rate'}
                                </label>
                                <span className="text-xs font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded">
                                    {erasureRate}%
                                </span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max="50"
                                step="5"
                                value={erasureRate}
                                onChange={(e) => setErasureRate(Number(e.target.value))}
                                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                            />
                            <div className="flex justify-between text-[10px] text-slate-400 mt-1 px-1">
                                <span>0% (Safe)</span>
                                <span>25%</span>
                                <span>50% (High Risk)</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex-1"></div>

                    <button
                        onClick={() => onStart({ scenarioId: selectedScenarioId, payload, erasureRate })}
                        className="w-full py-3.5 bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white rounded-xl font-bold tracking-wide shadow-lg shadow-indigo-200 transition-all transform hover:-translate-y-0.5 flex items-center justify-center gap-2"
                    >
                        <span>{locale === 'zh' ? '开始演示' : 'Start Demo'}</span>
                        <ArrowRight size={18} />
                    </button>
                </div>
            </motion.div>
        </div>
    );
};

export default WelcomeScreen;
