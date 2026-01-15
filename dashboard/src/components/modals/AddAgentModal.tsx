import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Plus, ChevronDown, ChevronUp, CheckCircle2, XCircle } from 'lucide-react';
import { useI18n } from '../../i18n/I18nContext';

interface AddAgentModalProps {
    isOpen: boolean;
    onClose: () => void;
    onApply: (data: { repoUrl: string; apiKey: string }) => void;
    apiKey: string;
    repoUrl: string;
}

const AddAgentModal: React.FC<AddAgentModalProps> = ({
    isOpen,
    onClose,
    onApply,
    apiKey,
    repoUrl
}) => {
    const { locale } = useI18n();
    const [localRepoUrl, setLocalRepoUrl] = React.useState(repoUrl);
    const [localApiKey, setLocalApiKey] = React.useState(apiKey);
    const [isCompatibilityOpen, setIsCompatibilityOpen] = useState(false);

    useEffect(() => {
        if (!isOpen) {
            return;
        }
        setLocalRepoUrl(repoUrl);
        setLocalApiKey(apiKey);
    }, [isOpen, repoUrl, apiKey]);

    const handleApply = () => {
        onApply({
            repoUrl: localRepoUrl.trim(),
            apiKey: localApiKey.trim()
        });
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                        className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm z-[100]"
                    />

                    <div
                        className="fixed z-[101]"
                        style={{
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)'
                        }}
                    >
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95, y: 10 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 10 }}
                            transition={{ type: "spring", duration: 0.4, bounce: 0.3 }}
                            className="w-[640px] max-w-[95vw] bg-white rounded-3xl shadow-2xl overflow-hidden border border-slate-100 max-h-[90vh] flex flex-col"
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Header */}
                            <div className="flex items-start justify-between px-8 pt-8 pb-4 bg-white shrink-0">
                                <div className="flex items-center gap-4">
                                    <div className="p-3 bg-indigo-50 rounded-2xl text-indigo-600">
                                        <Plus size={24} strokeWidth={2.5} />
                                    </div>
                                    <div>
                                        <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
                                            {locale === 'zh' ? '添加 Agent' : 'Add Your Agent'}
                                        </h2>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-slate-100 text-slate-600">
                                                {locale === 'zh' ? '自定义集成' : 'Custom Integration'}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                                <button
                                    onClick={onClose}
                                    className="p-2 -mr-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-xl transition-all"
                                >
                                    <X size={20} />
                                </button>
                            </div>

                            {/* Content */}
                            <div className="px-8 py-2 space-y-6 overflow-y-auto custom-scrollbar">

                                {/* Info Box */}
                                <div className="p-5 bg-indigo-50/80 rounded-2xl border border-indigo-100/50">
                                    <h3 className="text-sm font-bold text-indigo-900 mb-2 flex items-center gap-2">
                                        {locale === 'zh' ? '集成说明' : 'Integration Guide'}
                                    </h3>
                                    <p className="text-sm text-indigo-700/80 leading-relaxed">
                                        {locale === 'zh'
                                            ? '接入您的自定义 Agent 以进行评估。需要提供 GitHub 仓库地址和 API 密钥。'
                                            : 'Connect your custom agent for evaluation. Requires a GitHub repository URL and API key.'}
                                    </p>
                                </div>

                                <div className="space-y-6">
                                    <div>
                                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-slate-400"></div>
                                            {locale === 'zh' ? '仓库地址' : 'Repository URL'}
                                        </label>
                                        <div className="relative group">
                                            <input
                                                type="text"
                                                value={localRepoUrl}
                                                onChange={(e) => setLocalRepoUrl(e.target.value)}
                                                placeholder="https://github.com/your-org/your-agent"
                                                className="w-full px-4 py-3.5 bg-slate-50 rounded-2xl border-2 border-transparent focus:bg-white focus:border-indigo-100 focus:ring-4 focus:ring-indigo-50 outline-none transition-all text-sm font-mono text-slate-600 placeholder:text-slate-300"
                                            />
                                        </div>
                                    </div>

                                    <div>
                                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block flex items-center gap-2">
                                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
                                            {locale === 'zh' ? 'API 密钥' : 'API Key'}
                                        </label>
                                        <div className="relative group">
                                            <input
                                                type="password"
                                                value={localApiKey}
                                                onChange={(e) => setLocalApiKey(e.target.value)}
                                                placeholder="sk-..."
                                                className="w-full px-4 py-3.5 bg-slate-50 rounded-2xl border-2 border-transparent focus:bg-white focus:border-indigo-100 focus:ring-4 focus:ring-indigo-50 outline-none transition-all text-sm font-mono text-slate-600 placeholder:text-slate-300"
                                            />
                                        </div>
                                    </div>
                                </div>

                                {/* Compatibility Guide */}
                                <div className="border border-slate-200 rounded-2xl overflow-hidden">
                                    <button
                                        onClick={() => setIsCompatibilityOpen(!isCompatibilityOpen)}
                                        className="w-full px-5 py-3 bg-slate-50 flex items-center justify-between text-sm font-bold text-slate-700 hover:bg-slate-100 transition-colors"
                                    >
                                        <span>{locale === 'zh' ? '支持的 Agent 类型' : 'Supported Agent Types'}</span>
                                        {isCompatibilityOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                    </button>
                                    <AnimatePresence>
                                        {isCompatibilityOpen && (
                                            <motion.div
                                                initial={{ height: 0, opacity: 0 }}
                                                animate={{ height: 'auto', opacity: 1 }}
                                                exit={{ height: 0, opacity: 0 }}
                                                className="overflow-hidden"
                                            >
                                                <div className="p-5 bg-white border-t border-slate-200 space-y-4">
                                                    <div>
                                                        <h4 className="flex items-center gap-2 text-emerald-600 font-bold text-xs uppercase tracking-wider mb-2">
                                                            <CheckCircle2 size={14} />
                                                            {locale === 'zh' ? '完美适配 (One-Click)' : 'Generic Support (One-Click)'}
                                                        </h4>
                                                        <p className="text-xs text-slate-500 mb-2">
                                                            {locale === 'zh' ? '只要支持修改 OpenAI Base URL 即可：' : 'Any agent supporting OpenAI Base URL configuration:'}
                                                        </p>
                                                        <div className="grid grid-cols-2 gap-2">
                                                            {['AutoGPT', 'OpenAI Swarm', 'OpenAI Agents SDK', 'LangChain', 'LlamaIndex', 'CrewAI', 'MetaGPT', 'LiteLLM'].map(agent => (
                                                                <div key={agent} className="text-xs px-2 py-1 bg-emerald-50 text-emerald-700 rounded border border-emerald-100 text-center">
                                                                    {agent}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div>

                                                    <div className="pt-2 border-t border-slate-100">
                                                        <h4 className="flex items-center gap-2 text-rose-500 font-bold text-xs uppercase tracking-wider mb-2">
                                                            <XCircle size={14} />
                                                            {locale === 'zh' ? '暂不支持' : 'Not Supported'}
                                                        </h4>
                                                        <ul className="text-xs text-slate-500 space-y-1 list-disc pl-4">
                                                            <li>{locale === 'zh' ? '本地权重直接加载 (llama.cpp, torch)' : 'Local weights direct loading (llama.cpp, torch)'}</li>
                                                            <li>{locale === 'zh' ? 'SaaS 网页版 (ChatGPT Web, Claude Web)' : 'SaaS Web Interface (ChatGPT Web, Claude Web)'}</li>
                                                            <li>{locale === 'zh' ? '非 OpenAI 协议 (Google Gemini SDK)' : 'Non-OpenAI Protocols (Google Gemini SDK)'}</li>
                                                        </ul>
                                                    </div>
                                                </div>
                                            </motion.div>
                                        )}
                                    </AnimatePresence>
                                </div>
                            </div>

                            {/* Footer */}
                            <div className="p-8 pt-4 shrink-0 bg-white">
                                <div className="flex items-center gap-3">
                                    <button
                                        onClick={onClose}
                                        className="flex-1 py-3.5 rounded-xl text-slate-500 hover:bg-slate-50 hover:text-slate-700 transition-all font-bold text-sm"
                                    >
                                        {locale === 'zh' ? '取消' : 'Cancel'}
                                    </button>
                                    <button
                                        onClick={handleApply}
                                        className="flex-[2] py-3.5 rounded-xl bg-slate-900 text-white font-bold text-sm hover:bg-slate-800 hover:shadow-lg hover:shadow-slate-200 transition-all active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none"
                                        disabled={!localApiKey.trim() || !localRepoUrl.trim()}
                                    >
                                        {locale === 'zh' ? '连接 Agent' : 'Connect Agent'}
                                    </button>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </>
            )}
        </AnimatePresence>
    );
};

export default AddAgentModal;
