import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

interface SmileyAvatarProps {
  balance: number;
  size?: number;
  tier?: 'free' | 'quantum' | 'agent' | 'validator';
}

export const SmileyAvatar: React.FC<SmileyAvatarProps> = ({ balance, size = 200, tier = 'free' }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [mood, setMood] = useState<'ecstatic' | 'happy' | 'neutral' | 'concerned' | 'desperate'>('neutral');

  useEffect(() => {
    if (balance > 10000) setMood('ecstatic');
    else if (balance > 1000) setMood('happy');
    else if (balance > 100) setMood('neutral');
    else if (balance > 10) setMood('concerned');
    else setMood('desperate');
  }, [balance]);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.z = 4;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(size, size);
    renderer.setClearColor(0x000000, 0);
    mountRef.current.appendChild(renderer.domElement);

    // Lighting
    scene.add(new THREE.AmbientLight(0x404040, 2));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(2, 2, 3);
    scene.add(light);

    // Face color based on tier
    const colors: Record<string, number> = { free: 0xffd700, quantum: 0x00ff88, agent: 0xff6600, validator: 0xcc00ff };
    const faceColor = colors[tier] || 0xffd700;

    // Head
    const headGeo = new THREE.SphereGeometry(1, 32, 32);
    const headMat = new THREE.MeshStandardMaterial({ color: faceColor, roughness: 0.3, metalness: 0.1 });
    const head = new THREE.Mesh(headGeo, headMat);
    scene.add(head);

    // Eyes
    const eyeGeo = new THREE.SphereGeometry(0.15, 16, 16);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
    const pupilGeo = new THREE.SphereGeometry(0.08, 8, 8);
    const pupilMat = new THREE.MeshStandardMaterial({ color: 0x000000 });

    const leftEye = new THREE.Mesh(eyeGeo, eyeMat);
    leftEye.position.set(-0.3, 0.25, 0.9);
    const leftPupil = new THREE.Mesh(pupilGeo, pupilMat);
    leftPupil.position.set(-0.3, 0.25, 1.0);
    head.add(leftEye);
    head.add(leftPupil);

    const rightEye = new THREE.Mesh(eyeGeo, eyeMat);
    rightEye.position.set(0.3, 0.25, 0.9);
    const rightPupil = new THREE.Mesh(pupilGeo, pupilMat);
    rightPupil.position.set(0.3, 0.25, 1.0);
    head.add(rightEye);
    head.add(rightPupil);

    // Mouth (curve)
    const mouthGroup = new THREE.Group();
    const mouthCurve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(-0.3, -0.2, 0.9),
      new THREE.Vector3(0, -0.5, 0.9),
      new THREE.Vector3(0.3, -0.2, 0.9)
    );
    const mouthGeo = new THREE.TubeGeometry(mouthCurve, 20, 0.04, 8, false);
    const mouthMat = new THREE.MeshStandardMaterial({ color: 0x333333 });
    const mouth = new THREE.Mesh(mouthGeo, mouthMat);
    mouthGroup.add(mouth);
    head.add(mouthGroup);

    // Mood-based mouth adjustments
    const mouthAdjustments: Record<string, number> = {
      ecstatic: 0.4, happy: 0.15, neutral: 0, concerned: -0.15, desperate: -0.35
    };
    mouthGroup.position.y = mouthAdjustments[mood];
    mouthGroup.rotation.z = mood === 'desperate' ? Math.PI : 0;

    // Animation
    let frame = 0;
    const animate = () => {
      frame++;
      head.rotation.y = Math.sin(frame * 0.02) * 0.1;
      head.position.y = Math.sin(frame * 0.05) * (mood === 'ecstatic' ? 0.2 : 0.05);
      
      if (mood === 'ecstatic') {
        head.scale.setScalar(1 + Math.sin(frame * 0.1) * 0.05);
      }

      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };
    animate();

    return () => {
      mountRef.current?.removeChild(renderer.domElement);
      renderer.dispose();
      headGeo.dispose(); headMat.dispose();
      eyeGeo.dispose(); eyeMat.dispose();
      pupilGeo.dispose(); pupilMat.dispose();
      mouthGeo.dispose(); mouthMat.dispose();
    };
  }, [size, mood, tier]);

  return (
    <div 
      ref={mountRef} 
      style={{ width: size, height: size, cursor: 'pointer' }}
      title={`${mood.toUpperCase()} — ${balance.toLocaleString()} QUG`}
    />
  );
};
