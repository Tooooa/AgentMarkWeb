import React, { useRef, useEffect } from 'react';

import StepCard from '../execution/StepCard';
import type { Step } from '../../data/mockData';

import { Columns, Eye, EyeOff } from 'lucide-react';

interface ComparisonViewProps {
    visibleSteps: Step[];
    erasedIndices: Set<number>;
}

const ComparisonView: React.FC<ComparisonViewProps> = ({ visibleSteps, erasedIndices }) => {

    const scrollRefA = useRef<HTMLDivElement>(null);
    const scrollRefB = useRef<HTMLDivElement>(null);
    const bottomRefA = useRef<HTMLDivElement>(null);
    const bottomRefB = useRef<HTMLDivElement>(null);

    // Sync Scroll Logic (Simplified for auto-scroll)
    useEffect(() => {
        if (visibleSteps.length > 0) {
            bottomRefA.current?.scrollIntoView({ behavior: 'smooth' });
            bottomRefB.current?.scrollIntoView({ behavior: 'smooth' });
        }
    }, [visibleSteps]);

    // Manually handle scroll sync? For now, we just auto-scroll both.
    // Implementing true bi-directional sync scroll is complex without custom hooks, 
    // but the requirement says "Sync Scrolling".

    const handleScroll = (source: 'A' | 'B') => {
        const sourceRef = source === 'A' ? scrollRefA : scrollRefB;
        const targetRef = source === 'A' ? scrollRefB : scrollRefA;

        if (sourceRef.current && targetRef.current) {

            // Verify if we should sync (avoid loops) - rough implementation
            // targetRef.current.scrollTop = ratio * (targetRef.current.scrollHeight - targetRef.current.clientHeight);
            // Actually, since content is identical length mostly, direct scrollTop copy is better
            targetRef.current.scrollTop = sourceRef.current.scrollTop;
        }
    };

    return (
        <div className="h-full flex flex-col gap-4">
            <div className="flex items-center gap-2 px-4 py-2 bg-white rounded-xl shadow-sm border border-indigo-100 text-indigo-900">
                <Columns size={20} />
                <h2 className="font-bold">Comparison Mode (Contrast View)</h2>
                <div className="ml-auto flex gap-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
                    <span className="flex items-center gap-1"><EyeOff size={14} /> Base Model</span>
                    <span className="flex items-center gap-1"><Eye size={14} className="text-indigo-600" /> Watermarked</span>
                </div>
            </div>

            <div className="flex-1 grid grid-cols-2 gap-6 min-h-0">
                {/* Left: No Watermark */}
                <div className="flex flex-col gap-2 rounded-2xl bg-white border border-slate-200 overflow-hidden shadow-sm">
                    <div className="p-3 bg-slate-50 border-b border-slate-100 text-center font-bold text-slate-600 text-sm">
                        Original Model (No Watermark)
                    </div>
                    <div
                        className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-200"
                        ref={scrollRefA}
                        onScroll={() => handleScroll('A')}
                    >
                        {visibleSteps.map((step) => (
                            <StepCard
                                key={`nw-${step.stepIndex}`}
                                step={step}
                                isErased={false} // Assume no erasure simulation on base model for clarity? OR same erasure. Let's assume perfect transmission for base as control? Or same environment. Let's use false for now to show "clean" output.
                                showWatermarkDetails={false}
                            />
                        ))}
                        <div ref={bottomRefA} className="h-4" />
                    </div>
                </div>

                {/* Right: Watermarked */}
                <div className="flex flex-col gap-2 rounded-2xl bg-white border border-indigo-200 overflow-hidden shadow-md ring-1 ring-indigo-50">
                    <div className="p-3 bg-indigo-50 border-b border-indigo-100 text-center font-bold text-indigo-700 text-sm flex items-center justify-center gap-2">
                        Ours (Watermarked)
                    </div>
                    <div
                        className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-indigo-100"
                        ref={scrollRefB}
                        onScroll={() => handleScroll('B')}
                    >
                        {visibleSteps.map((step) => (
                            <StepCard
                                key={`wm-${step.stepIndex}`}
                                step={step}
                                isErased={erasedIndices.has(step.stepIndex)}
                                showWatermarkDetails={true}
                            />
                        ))}
                        <div ref={bottomRefB} className="h-4" />
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ComparisonView;
