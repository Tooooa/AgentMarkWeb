import React, { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment, Float, PresentationControls, useCursor, useTexture } from '@react-three/drei';
import * as THREE from 'three';

// Configuration
const PAGE_WIDTH = 3;
const PAGE_HEIGHT = 4.2;
const PAGE_DEPTH = 0.04; // Slightly thicker for better presence
const CORNER_RADIUS = 0.15; // Larger radius for more visible rounded corners
const ROUGHNESS = 0.15; // Smoother surface
const METALNESS = 0.05; // Slight metallic sheen

interface PageProps {
    number: number;
    opened: boolean;
    totalPages: number;
    frontTexture?: THREE.Texture | null;
    backTexture?: THREE.Texture | null;
    isLeftPage?: boolean;
    currentSpread: number;
    visible?: boolean;
}

function Page({ number, opened, totalPages, frontTexture, backTexture, isLeftPage = false, currentSpread, visible = true }: PageProps) {
    const group = useRef<THREE.Group>(null);
    const meshRef = useRef<THREE.Mesh>(null);

    const zRef = useRef(0);
    const opacityRef = useRef(visible ? 1 : 0);
    
    useFrame((_, delta) => {
        if (!group.current) return;

        // All pages rotate around the spine (x=0)
        const targetRotation = opened ? -Math.PI : 0;
        
        const rotDiff = targetRotation - group.current.rotation.y;
        
        // Smooth easing animation
        const easing = 1 - Math.pow(1 - 0.08, delta * 60);
        group.current.rotation.y += rotDiff * easing;

        const progress = Math.max(0, Math.min(1, -group.current.rotation.y / Math.PI));
        
        // Z position - simpler calculation
        const spacing = 0.008;
        const frontZ = 0.5;
        
        let targetZ: number;
        if (number === currentSpread || (opened && number === currentSpread - 1)) {
            // Current visible pages at front
            targetZ = frontZ;
        } else if (number < currentSpread) {
            // Already flipped pages - stack behind
            targetZ = frontZ - (currentSpread - number) * spacing;
        } else {
            // Not yet flipped pages - stack behind
            targetZ = frontZ - (number - currentSpread) * spacing;
        }
        
        // Smoothly animate Z position
        zRef.current += (targetZ - zRef.current) * easing;

        // Lift curve during animation
        const liftHeight = 1.5;
        const lift = Math.sin(progress * Math.PI) * liftHeight;

        group.current.position.z = zRef.current + lift;
    });

    const isCover = number === 0;

    // Create rounded corner mask
    const roundedMask = useMemo(() => {
        const canvas = document.createElement('canvas');
        const size = 1024;
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        const r = (CORNER_RADIUS / PAGE_WIDTH) * size;
        
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, size, size);
        
        ctx.fillStyle = 'white';
        ctx.beginPath();
        ctx.moveTo(r, 0);
        ctx.lineTo(size - r, 0);
        ctx.quadraticCurveTo(size, 0, size, r);
        ctx.lineTo(size, size - r);
        ctx.quadraticCurveTo(size, size, size - r, size);
        ctx.lineTo(r, size);
        ctx.quadraticCurveTo(0, size, 0, size - r);
        ctx.lineTo(0, r);
        ctx.quadraticCurveTo(0, 0, r, 0);
        ctx.closePath();
        ctx.fill();

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }, []);

    // Define materials array
    const materialArray = useMemo(() => {
        // Edge material - brown color for paper edges
        const sideMat = new THREE.MeshStandardMaterial({ 
            color: '#8B7355',
            roughness: 0.4,
            metalness: 0.0
        });

        // Paper color - warm brown/tan
        const paperColor = '#D2B48C';

        // FRONT face
        let frontMat;
        if (frontTexture) {
            frontMat = new THREE.MeshStandardMaterial({
                map: frontTexture,
                alphaMap: roundedMask,
                transparent: true,
                roughness: isCover ? 0.2 : ROUGHNESS,
                metalness: isCover ? 0.1 : METALNESS,
                color: '#ffffff'
            });
        } else {
            frontMat = new THREE.MeshStandardMaterial({
                color: paperColor,
                alphaMap: roundedMask,
                transparent: true,
                roughness: ROUGHNESS,
                metalness: METALNESS
            });
        }

        // BACK face
        let backMat;
        if (backTexture) {
            backMat = new THREE.MeshStandardMaterial({
                map: backTexture,
                alphaMap: roundedMask,
                transparent: true,
                roughness: ROUGHNESS,
                metalness: METALNESS,
                color: '#ffffff'
            });
        } else {
            backMat = new THREE.MeshStandardMaterial({ 
                color: paperColor,
                alphaMap: roundedMask,
                transparent: true,
                roughness: ROUGHNESS,
                metalness: METALNESS
            });
        }

        return [sideMat, sideMat, sideMat, sideMat, frontMat, backMat];
    }, [isCover, frontTexture, backTexture, roundedMask]);

    return (
        <group ref={group}>
            {/* All pages pivot from spine (x=0), positioned to the right */}
            <group position={[PAGE_WIDTH / 2, 0, 0]}>
                {/* Main page */}
                <mesh ref={meshRef} material={materialArray} castShadow receiveShadow>
                    <boxGeometry args={[PAGE_WIDTH, PAGE_HEIGHT, isCover ? PAGE_DEPTH * 1.5 : PAGE_DEPTH]} />
                </mesh>
            </group>
        </group>
    );
}

const Book = ({ pageIndex, setTotalPages, setCurrentPage }: { pageIndex: number, setTotalPages: (n: number) => void, setCurrentPage: (n: number) => void }) => {
    const [hovered, setHover] = useState(false);
    useCursor(hovered);

    // Load cover
    const coverTexture = useTexture('/book_cover.png');
    coverTexture.colorSpace = THREE.SRGBColorSpace;

    // Load all spread textures (01-10)
    const spreadTextures = useTexture([
        '/book_spread_01.png',
        '/book_spread_02.png',
        '/book_spread_03.png',
        '/book_spread_04.png',
        '/book_spread_05.png',
        '/book_spread_06.png',
        '/book_spread_07.png',
        '/book_spread_08.png',
        '/book_spread_09.png',
        '/book_spread_10.png',
    ]);
    
    // Set color space for all spreads
    spreadTextures.forEach(tex => {
        tex.colorSpace = THREE.SRGBColorSpace;
    });

    // Load last page
    const lastTexture = useTexture('/book_last.png');
    lastTexture.colorSpace = THREE.SRGBColorSpace;

    // Split each spread into left and right halves (including last page)
    const spreads = useMemo(() => {
        const allSpreads = [...spreadTextures, lastTexture].map(spreadTexture => {
            const left = spreadTexture.clone();
            left.colorSpace = THREE.SRGBColorSpace;
            left.offset.set(0, 0);
            left.repeat.set(0.5, 1);
            left.needsUpdate = true;

            const right = spreadTexture.clone();
            right.colorSpace = THREE.SRGBColorSpace;
            right.offset.set(0.5, 0);
            right.repeat.set(0.5, 1);
            right.needsUpdate = true;

            return { left, right };
        });
        return allSpreads;
    }, [spreadTextures, lastTexture]);

    // Total pages: cover + 11 spread right pages = 12 pages
    const totalPages = 12;
    
    useEffect(() => {
        setTotalPages(11); // 12 navigation points: 0=cover, 1-11=spreads
        setCurrentPage(pageIndex);
    }, [pageIndex, setTotalPages, setCurrentPage]);

    // Build all pages
    const allPages = [];
    
    // Cover page (page 0)
    allPages.push(
        <Page
            key={0}
            number={0}
            totalPages={totalPages}
            opened={pageIndex >= 1}
            frontTexture={coverTexture}
            backTexture={spreads[0].left}
            isLeftPage={false}
            currentSpread={pageIndex}
            visible={true}
        />
    );
    
    // Spread pages
    spreads.forEach((spread, spreadIndex) => {
        const pageNumber = spreadIndex + 1;
        const spreadNumber = spreadIndex + 1;
        const nextSpread = spreads[spreadIndex + 1];
        
        allPages.push(
            <Page
                key={pageNumber}
                number={pageNumber}
                totalPages={totalPages}
                opened={pageIndex > spreadNumber}
                frontTexture={spread.right}
                backTexture={nextSpread ? nextSpread.left : null}
                isLeftPage={false}
                currentSpread={pageIndex}
                visible={true}
            />
        );
    });

    return (
        <group
            onPointerOver={(e) => { e.stopPropagation(); setHover(true); }}
            onPointerOut={(e) => { e.stopPropagation(); setHover(false); }}
        >
            {allPages}
        </group>
    );
};

export default function BookDemo({ onBack }: { onBack: () => void }) {
    const [totalPages, setTotalPages] = useState(0);
    const [currentPage, setCurrentPage] = useState(0);

    const handleNext = () => {
        if (currentPage < totalPages) {
            setCurrentPage(prev => prev + 1);
        }
    };

    const handlePrev = () => {
        if (currentPage > 0) {
            setCurrentPage(prev => prev - 1);
        }
    };

    return (
        <div className="w-full h-screen bg-sky-100 relative overflow-hidden font-sans">
            {/* Light blue background with subtle gradients */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] bg-sky-200/50 rounded-full blur-3xl" />
                <div className="absolute top-[40%] -right-[10%] w-[40%] h-[60%] bg-blue-200/40 rounded-full blur-3xl" />
            </div>

            <button
                onClick={onBack}
                className="absolute top-8 left-8 z-50 text-sky-600 hover:text-sky-800 transition-colors flex items-center gap-2 font-medium"
            >
                ← Back
            </button>

            {/* Enhanced page indicator */}
            <div className="absolute bottom-12 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-white/70 backdrop-blur-lg px-8 py-4 rounded-full shadow-lg border border-sky-200/60 transition-all duration-300">
                {new Array(totalPages + 1).fill(0).map((_, i) => (
                    <div
                        key={i}
                        onClick={() => setCurrentPage(i)}
                        className={`
                            rounded-full transition-all duration-300 cursor-pointer
                            ${i === currentPage
                                ? 'w-3 h-3 bg-gradient-to-r from-sky-400 to-blue-500 scale-125 shadow-lg shadow-sky-500/50'
                                : 'w-2 h-2 bg-sky-300 hover:bg-sky-400 hover:scale-110'
                            }
                        `}
                    />
                ))}
            </div>
            
            {/* Invisible click areas for navigation */}
            <div 
                className="absolute left-0 top-0 w-1/2 h-full z-40 cursor-pointer group"
                onClick={handlePrev}
                title="Previous Page"
            >
                {/* Visual indicator on hover */}
                <div className="absolute left-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    <div className="bg-white/90 backdrop-blur-md rounded-full p-3 shadow-lg border border-sky-200">
                        <svg className="w-8 h-8 text-sky-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                    </div>
                </div>
            </div>
            <div 
                className="absolute right-0 top-0 w-1/2 h-full z-40 cursor-pointer group"
                onClick={handleNext}
                title="Next Page"
            >
                {/* Visual indicator on hover */}
                <div className="absolute right-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                    <div className="bg-white/90 backdrop-blur-md rounded-full p-3 shadow-lg border border-sky-200">
                        <svg className="w-8 h-8 text-sky-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </div>
                </div>
            </div>

            <Canvas shadows camera={{ position: [0, 0, 8.5], fov: 35 }}>
                {/* Enhanced lighting setup for premium look */}
                <ambientLight intensity={0.6} />
                
                {/* Main directional light with soft shadows */}
                <directionalLight 
                    position={[5, 8, 5]} 
                    intensity={1.5} 
                    castShadow
                    shadow-mapSize-width={4096}
                    shadow-mapSize-height={4096}
                    shadow-camera-far={50}
                    shadow-camera-left={-10}
                    shadow-camera-right={10}
                    shadow-camera-top={10}
                    shadow-camera-bottom={-10}
                    shadow-bias={-0.0001}
                />
                
                {/* Fill lights for softer shadows */}
                <pointLight position={[-5, 5, -5]} intensity={0.4} color="#e0e7ff" />
                <pointLight position={[5, -3, 5]} intensity={0.3} color="#fef3c7" />
                
                {/* Rim light for edge definition */}
                <spotLight 
                    position={[0, 10, -5]} 
                    angle={0.4} 
                    penumbra={1} 
                    intensity={0.6} 
                    color="#ffffff"
                    castShadow
                />

                <PresentationControls
                    global
                    rotation={[0, 0, 0]}
                    polar={[-Math.PI / 12, Math.PI / 12]}
                    azimuth={[-Math.PI / 6, Math.PI / 6]}
                >
                    <Float
                        rotationIntensity={0.1}
                        floatIntensity={0.2}
                        speed={1.2}
                        floatingRange={[-0.03, 0.03]}
                    >
                        <React.Suspense fallback={null}>
                            <Book pageIndex={currentPage} setTotalPages={setTotalPages} setCurrentPage={setCurrentPage} />
                        </React.Suspense>
                    </Float>
                </PresentationControls>

                {/* Premium environment with soft lighting */}
                <Environment preset="apartment" />
                
                {/* Subtle fog for depth */}
                <fog attach="fog" args={['#e0f2fe', 12, 25]} />
            </Canvas>
        </div>
    );
}
