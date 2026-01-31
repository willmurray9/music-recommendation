'use client';

import { useRef, useMemo, useState, useEffect, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsType } from 'three-stdlib';

interface Track {
  id: string;
  name: string;
  artist: string;
  genres: string[];
  popularity: number;
}

interface Visualization3DProps {
  coords: number[][];
  tracks: Track[];
  selectedIndices?: number[];
  recommendedIndices?: number[];
  onTrackSelect?: (index: number) => void;
  colorBy?: 'popularity' | 'genre';
}

// Genre color mapping
const GENRE_COLORS: Record<string, string> = {
  'hip hop': '#ff6b6b',
  'rap': '#ff8e8e',
  'pop': '#4ecdc4',
  'rock': '#45b7d1',
  'country': '#96ceb4',
  'edm': '#dda0dd',
  'r&b': '#ffeaa7',
  'metal': '#636e72',
  'indie': '#fd79a8',
  'jazz': '#a29bfe',
  'classical': '#74b9ff',
  'reggae': '#00b894',
  'latin': '#e17055',
  'worship': '#81ecec',
  'default': '#b2bec3',
};

function getGenreColor(genres: string[]): THREE.Color {
  for (const genre of genres) {
    const lowerGenre = genre.toLowerCase();
    for (const [key, color] of Object.entries(GENRE_COLORS)) {
      if (lowerGenre.includes(key)) {
        return new THREE.Color(color);
      }
    }
  }
  return new THREE.Color(GENRE_COLORS.default);
}

function getPopularityColor(popularity: number): THREE.Color {
  // Blue (cold/obscure) to Red (hot/popular)
  const hue = (1 - popularity / 100) * 0.6; // 0.6 (blue) to 0 (red)
  return new THREE.Color().setHSL(hue, 0.8, 0.5);
}

// Camera controller component for smooth zooming
function CameraController({ 
  targetPosition, 
  controlsRef 
}: { 
  targetPosition: THREE.Vector3 | null;
  controlsRef: React.RefObject<OrbitControlsType>;
}) {
  const { camera } = useThree();
  const isAnimating = useRef(false);
  const animationProgress = useRef(0);
  const startPosition = useRef(new THREE.Vector3());
  const startTarget = useRef(new THREE.Vector3());

  useEffect(() => {
    if (targetPosition && controlsRef.current) {
      // Start animation
      isAnimating.current = true;
      animationProgress.current = 0;
      startPosition.current.copy(camera.position);
      startTarget.current.copy(controlsRef.current.target);
    }
  }, [targetPosition, camera, controlsRef]);

  useFrame(() => {
    if (!isAnimating.current || !targetPosition || !controlsRef.current) return;

    animationProgress.current += 0.02; // Animation speed
    const t = Math.min(animationProgress.current, 1);
    // Ease out cubic
    const eased = 1 - Math.pow(1 - t, 3);

    // Calculate new camera position (offset from target)
    const endPosition = new THREE.Vector3(
      targetPosition.x + 15,
      targetPosition.y + 10,
      targetPosition.z + 15
    );

    // Interpolate camera position
    camera.position.lerpVectors(startPosition.current, endPosition, eased);
    
    // Interpolate controls target
    controlsRef.current.target.lerpVectors(startTarget.current, targetPosition, eased);
    controlsRef.current.update();

    if (t >= 1) {
      isAnimating.current = false;
    }
  });

  return null;
}

// Point cloud component for efficient rendering of 62K points
function PointCloud({
  coords,
  tracks,
  selectedIndices = [],
  recommendedIndices = [],
  onTrackSelect,
  colorBy = 'popularity',
  controlsRef,
  onCameraTarget,
}: Visualization3DProps & { 
  controlsRef: React.RefObject<OrbitControlsType>;
  onCameraTarget: (pos: THREE.Vector3) => void;
}) {
  const pointsRef = useRef<THREE.Points>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const { camera, raycaster, pointer } = useThree();
  const prevSelectedLength = useRef(selectedIndices.length);

  // Auto-zoom when a new seed is added
  useEffect(() => {
    if (selectedIndices.length > prevSelectedLength.current && selectedIndices.length > 0) {
      // New seed was added, zoom to the latest one
      const latestIdx = selectedIndices[selectedIndices.length - 1];
      const targetPos = new THREE.Vector3(
        coords[latestIdx][0] * 50,
        coords[latestIdx][1] * 50,
        coords[latestIdx][2] * 50
      );
      onCameraTarget(targetPos);
    }
    prevSelectedLength.current = selectedIndices.length;
  }, [selectedIndices, coords, onCameraTarget]);

  // Create geometry with positions and colors
  const { positions, colors, selectedPositions, recommendedPositions } = useMemo(() => {
    const positions = new Float32Array(coords.length * 3);
    const colors = new Float32Array(coords.length * 3);
    
    for (let i = 0; i < coords.length; i++) {
      positions[i * 3] = coords[i][0] * 50;
      positions[i * 3 + 1] = coords[i][1] * 50;
      positions[i * 3 + 2] = coords[i][2] * 50;
      
      const track = tracks[i];
      const color = colorBy === 'genre' 
        ? getGenreColor(track?.genres || [])
        : getPopularityColor(track?.popularity || 50);
      
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }

    // Selected track positions (for highlighting)
    const selectedPositions = new Float32Array(selectedIndices.length * 3);
    selectedIndices.forEach((idx, i) => {
      selectedPositions[i * 3] = coords[idx][0] * 50;
      selectedPositions[i * 3 + 1] = coords[idx][1] * 50;
      selectedPositions[i * 3 + 2] = coords[idx][2] * 50;
    });

    // Recommended track positions
    const recommendedPositions = new Float32Array(recommendedIndices.length * 3);
    recommendedIndices.forEach((idx, i) => {
      recommendedPositions[i * 3] = coords[idx][0] * 50;
      recommendedPositions[i * 3 + 1] = coords[idx][1] * 50;
      recommendedPositions[i * 3 + 2] = coords[idx][2] * 50;
    });

    return { positions, colors, selectedPositions, recommendedPositions };
  }, [coords, tracks, selectedIndices, recommendedIndices, colorBy]);

  // Raycasting for hover/click
  useFrame(() => {
    if (!pointsRef.current) return;
    
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(pointsRef.current);
    
    if (intersects.length > 0) {
      const idx = intersects[0].index;
      if (idx !== undefined && idx !== hovered) {
        setHovered(idx);
      }
    } else if (hovered !== null) {
      setHovered(null);
    }
  });

  const handleClick = () => {
    if (hovered !== null && onTrackSelect) {
      onTrackSelect(hovered);
    }
  };

  return (
    <group onClick={handleClick}>
      {/* Main point cloud */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={positions.length / 3}
            array={positions}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={colors.length / 3}
            array={colors}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.5}
          vertexColors
          transparent
          opacity={0.6}
          sizeAttenuation
        />
      </points>

      {/* Selected tracks (larger, highlighted) */}
      {selectedPositions.length > 0 && (
        <points>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={selectedPositions.length / 3}
              array={selectedPositions}
              itemSize={3}
            />
          </bufferGeometry>
          <pointsMaterial size={3} color="#1DB954" />
        </points>
      )}

      {/* Recommended tracks */}
      {recommendedPositions.length > 0 && (
        <points>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={recommendedPositions.length / 3}
              array={recommendedPositions}
              itemSize={3}
            />
          </bufferGeometry>
          <pointsMaterial size={2} color="#FFD700" />
        </points>
      )}

      {/* Hover tooltip */}
      {hovered !== null && tracks[hovered] && (
        <Html
          position={[
            coords[hovered][0] * 50,
            coords[hovered][1] * 50 + 2,
            coords[hovered][2] * 50,
          ]}
        >
          <div className="bg-black/90 text-white px-3 py-2 rounded-lg text-sm whitespace-nowrap pointer-events-none">
            <div className="font-semibold">{tracks[hovered].name}</div>
            <div className="text-gray-300">{tracks[hovered].artist}</div>
          </div>
        </Html>
      )}
    </group>
  );
}

function Scene(props: Visualization3DProps) {
  const controlsRef = useRef<OrbitControlsType>(null);
  const [cameraTarget, setCameraTarget] = useState<THREE.Vector3 | null>(null);

  const handleCameraTarget = useCallback((pos: THREE.Vector3) => {
    setCameraTarget(pos);
  }, []);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      <PointCloud 
        {...props} 
        controlsRef={controlsRef}
        onCameraTarget={handleCameraTarget}
      />
      <CameraController 
        targetPosition={cameraTarget} 
        controlsRef={controlsRef} 
      />
      <OrbitControls
        ref={controlsRef}
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={200}
      />
    </>
  );
}

export default function Visualization3D(props: Visualization3DProps) {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <div className="w-full h-full bg-spotify-dark flex items-center justify-center">
        <div className="text-spotify-light">Loading visualization...</div>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gradient-to-b from-spotify-black to-spotify-dark">
      <Canvas
        camera={{ position: [0, 0, 100], fov: 60 }}
        gl={{ antialias: true }}
      >
        <Scene {...props} />
      </Canvas>
    </div>
  );
}
