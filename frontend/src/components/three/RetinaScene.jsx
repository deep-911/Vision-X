import { Canvas, useFrame } from "@react-three/fiber";
import {
  Float,
  Line,
  MeshTransmissionMaterial,
  OrbitControls,
} from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

// Clean, Hospital/Clinical Palette (No neon AI colors)
const COLORS = {
  shell: "#FFF8F1", // Clean clear glass
  shellEdge: "#D8B79E",

  retinalSurface: "#C96A3D", // Deep slate background to make vessels pop clearly

  vessel: "#6F231F", // Surgical Teal
  vesselSecondary: "#8F3228", // Darker Teal
  vesselFine: "#A64E3B", // Slate gray for subtle fine vessels

  scanner: "#A8745C", // Clean slate metal
  scannerBright: "#C98A55", // Clinical success green

  highlight: "#F3C36A", // Medical amber/gold for highlights

  opticDisc: "#F4D6A3", // Anatomical amber
  fovea: "#7C2A24", // Clean clinical red

  signal: "#F1C38C", // Soft white/slate indicators
};

function projectToSphere(x, y, radius = 1.5) {
  const limit = radius * 0.96;

  const px = THREE.MathUtils.clamp(
    x,
    -limit,
    limit
  );

  const py = THREE.MathUtils.clamp(
    y,
    -limit,
    limit
  );

  const distance =
    px * px + py * py;

  const z = Math.sqrt(
    Math.max(
      0.04,
      radius * radius - distance
    )
  );

  return [px, py, z];
}

function createVessel(
  angle,
  length,
  startRadius,
  bend,
  radius = 1.5
) {
  return Array.from(
    { length: 48 },
    (_, i) => {
      const t = i / 47;

      const distance =
        startRadius +
        t * length;

      const offset =
        Math.sin(
          t * Math.PI * 2.4
        ) *
        bend *
        (1 - t);

      const x =
        Math.cos(angle) *
          distance -
        Math.sin(angle) *
          offset;

      const y =
        Math.sin(angle) *
          distance +
        Math.cos(angle) *
          offset;

      const point =
        projectToSphere(
          x,
          y,
          radius
        );

      return [
        point[0],
        point[1],
        point[2] +
          0.015 +
          Math.sin(t * Math.PI) *
            0.025,
      ];
    }
  );
}

function RetinalVessels() {
  const vessels = useMemo(() => {
    const primary = [
      [0.05, 1.45, 1.02, 0.05],
      [0.53, 1.32, 0.92, 0.06],
      [1.05, 1.28, 0.87, 0.05],
      [1.63, 1.25, 0.91, 0.05],
      [2.15, 1.28, 0.90, 0.05],
      [2.72, 1.32, 0.96, 0.05],
      [3.26, 1.35, 0.92, 0.05],
      [3.81, 1.28, 0.85, 0.05],
      [4.42, 1.23, 0.83, 0.04],
      [5.02, 1.30, 0.91, 0.05],
      [5.55, 1.34, 0.93, 0.05],
    ];

    const secondary = [
      [0.23, 0.88, 0.70, 0.035],
      [0.77, 0.87, 0.65, 0.035],
      [1.28, 0.91, 0.62, 0.04],
      [1.83, 0.86, 0.68, 0.035],
      [2.39, 0.91, 0.64, 0.035],
      [2.94, 0.88, 0.66, 0.04],
      [3.48, 0.89, 0.62, 0.035],
      [4.04, 0.87, 0.66, 0.035],
      [4.60, 0.89, 0.64, 0.035],
      [5.15, 0.87, 0.67, 0.035],
    ];

    return [
      ...primary.map(
        ([angle, start, length, bend]) => ({
          points: createVessel(
            angle,
            length,
            start,
            bend,
            1.5
          ),
          primary: true,
        })
      ),

      ...secondary.map(
        ([angle, start, length, bend]) => ({
          points: createVessel(
            angle,
            length,
            start,
            bend,
            1.495
          ),
          primary: false,
        })
      ),
    ];
  }, []);

  return (
    <group>
      {vessels.map(
        (vessel, index) => (
          <Line
            key={index}
            points={vessel.points}
            color={
              vessel.primary
                ? COLORS.vessel
                : COLORS.vesselSecondary
            }
            lineWidth={
              vessel.primary
                ? 2.5
                : 1.2
            }
            transparent
            opacity={
              vessel.primary
                ? 0.9
                : 0.5
            }
          />
        )
      )}
    </group>
  );
}

function FineVessels() {
  const lines = useMemo(() => {
    return Array.from(
      { length: 20 },
      (_, i) => {
        const angle =
          (Math.PI * 2 * i) /
          20;

        return createVessel(
          angle,
          0.5 +
            (i % 3) * 0.07,
          0.76,
          0.02,
          1.49
        );
      }
    );
  }, []);

  return (
    <group>
      {lines.map(
        (points, index) => (
          <Line
            key={index}
            points={points}
            color={COLORS.vesselFine}
            lineWidth={0.6}
            transparent
            opacity={0.3}
          />
        )
      )}
    </group>
  );
}

function RetinalSignals() {
  const points = useMemo(() => {
    return Array.from(
      { length: 72 },
      (_, i) => {
        const phi =
          Math.acos(
            1 -
              (2 * (i + 1)) /
                73
          );

        const theta =
          Math.sqrt(73 * Math.PI) *
          phi;

        const radius =
          1.52 +
          (i % 5) * 0.018;

        return [
          radius *
            Math.cos(theta) *
            Math.sin(phi),

          radius *
            Math.sin(theta) *
            Math.sin(phi),

          radius *
            Math.cos(phi),
        ];
      }
    );
  }, []);

  return (
    <group>
      {points.map(
        (position, index) => (
          <mesh
            key={index}
            position={[
              position[0],
              position[1],
              position[2] +
                0.02,
            ]}
            scale={
              index % 9 === 0
                ? 1.65
                : 0.68
            }
          >
            <sphereGeometry
              args={[
                0.011,
                10,
                10,
              ]}
            />

            <meshStandardMaterial
              color={
                index % 9 === 0
                  ? COLORS.highlight
                  : COLORS.signal
              }
              emissive={
                index % 9 === 0
                  ? COLORS.highlight
                  : COLORS.signal
              }
              emissiveIntensity={
                index % 9 === 0
                  ? 0.5
                  : 0.1
              }
              transparent
              opacity={
                index % 9 === 0
                  ? 0.9
                  : 0.32
              }
            />
          </mesh>
        )
      )}
    </group>
  );
}

function CentralAnatomy() {
  const pulse = useRef(null);

  useFrame((state) => {
    if (!pulse.current) return;

    const scale =
      1 +
      Math.sin(
        state.clock.elapsedTime *
          1.4
      ) *
        0.045;

    pulse.current.scale.set(
      scale,
      scale,
      scale
    );
  });

  return (
    <group>
      {/* optic disc */}

      <mesh
        position={[
          0.30,
          0.12,
          1.53,
        ]}
      >
        <sphereGeometry
          args={[
            0.19,
            32,
            32,
          ]}
        />

        <meshStandardMaterial
          color={COLORS.opticDisc}
          roughness={0.8}
          transparent
          opacity={0.35}
        />
      </mesh>

      {/* fovea */}

      <group
        ref={pulse}
        position={[
          -0.14,
          -0.07,
          1.54,
        ]}
      >
        <mesh>
          <torusGeometry
            args={[
              0.073,
              0.009,
              16,
              48,
            ]}
          />

          <meshBasicMaterial
            color={COLORS.fovea}
            transparent
            opacity={0.75}
          />
        </mesh>

        <mesh>
          <sphereGeometry
            args={[
              0.019,
              16,
              16,
            ]}
          />

          <meshBasicMaterial
            color={COLORS.fovea}
          />
        </mesh>
      </group>
    </group>
  );
}

function RetinaBody() {
  return (
    <group>

      {/* outer dark imaging shell */}

      <mesh>
        <sphereGeometry
          args={[
            1.72,
            64,
            64,
          ]}
        />

        <MeshTransmissionMaterial
          backside
          samples={4}
          thickness={0.55}
          chromaticAberration={0.02}
          anisotropy={0.1}
          distortion={0.05}
          distortionScale={0.1}
          temporalDistortion={0.02}
          color={COLORS.shell}
          transmission={0.95} // Increased transmission for better glass look
          roughness={0.1} // Reduced roughness to remove the murky frosted look
          transparent
          opacity={1}
        />
      </mesh>

      {/* dark retinal body */}

      <mesh>
        <sphereGeometry
          args={[
            1.51,
            64,
            64,
          ]}
        />

        <meshStandardMaterial
          color={
            COLORS.retinalSurface
          }
          roughness={0.9}
          metalness={0.1}
          transparent
          opacity={0.92} // Warm fundus field
        />
      </mesh>

      {/* anatomical grid */}

      <mesh>
        <sphereGeometry
          args={[
            1.525,
            34,
            34,
          ]}
        />

        <meshStandardMaterial
          color="#8E4E36" // Warm retinal structural grid
          wireframe
          transparent
          opacity={0.15}
        />
      </mesh>

      <RetinalVessels />
      <FineVessels />
      <RetinalSignals />
      <CentralAnatomy />
    </group>
  );
}

function ScannerAssembly() {
  const outer = useRef(null);
  const inner = useRef(null);

  useFrame((state) => {
    if (outer.current) {
      outer.current.rotation.z =
        state.clock.elapsedTime *
        0.16;
    }

    if (inner.current) {
      inner.current.rotation.z =
        -state.clock.elapsedTime *
        0.11;
    }
  });

  return (
    <group>

      <group ref={outer}>

        {/* primary scanner */}

        <mesh
          rotation={[
            Math.PI / 2,
            0.08,
            0,
          ]}
        >
          <torusGeometry
            args={[
              1.94,
              0.009,
              16,
              180,
            ]}
          />

          <meshBasicMaterial
            color={COLORS.scanner}
            transparent
            opacity={0.4}
          />
        </mesh>

        {/* scanning segment */}

        <mesh
          rotation={[
            Math.PI / 2,
            0.2,
            0,
          ]}
        >
          <torusGeometry
            args={[
              2.12,
              0.014,
              16,
              180,
              Math.PI * 0.33,
            ]}
          />

          <meshStandardMaterial
            color={
              COLORS.scannerBright
            }
            emissive={
              COLORS.scannerBright
            }
            emissiveIntensity={0.5}
            transparent
            opacity={0.9}
          />
        </mesh>

      </group>

      <group ref={inner}>

        <mesh
          rotation={[
            Math.PI / 2,
            -0.3,
            0,
          ]}
        >
          <torusGeometry
            args={[
              2.34,
              0.004,
              12,
              180,
            ]}
          />

          <meshBasicMaterial
            color="#64748B"
            transparent
            opacity={0.3}
          />
        </mesh>

      </group>
    </group>
  );
}

function Scene() {
  const group = useRef(null);

  useFrame((state) => {
    if (!group.current) return;

    group.current.rotation.y =
      Math.sin(
        state.clock.elapsedTime *
          0.13
      ) *
      0.10;

    group.current.rotation.x =
      Math.sin(
        state.clock.elapsedTime *
          0.18
      ) *
      0.025;
  });

  return (
    <group ref={group}>
      <RetinaBody />
      <ScannerAssembly />
    </group>
  );
}

export default function RetinaScene() {
  return (
    <div className="absolute inset-0">

      <Canvas
        camera={{
          position: [
            0,
            0.05,
            5.8,
          ],
          fov: 38,
        }}
        dpr={[1, 2]} // Slightly sharper resolution
      >

        {/* Matched to the UI slate-50 background */}
        <color
          attach="background"
          args={[
            "#F8FAFC",
          ]}
        />

        {/* Adjusted lighting to prevent blown-out white spots */}
        <ambientLight
          intensity={0.8}
        />

        <directionalLight
          position={[
            4,
            5,
            6,
          ]}
          intensity={1.2}
          color="#FFFFFF"
        />

        <directionalLight
          position={[
            -4,
            -3,
            2,
          ]}
          intensity={0.8}
          color="#FFE2C2" // Warm fundus fill light
        />

        <pointLight
          position={[
            1,
            1,
            4,
          ]}
          intensity={0.6}
          color="#FFFFFF"
        />

        <Float
          speed={0.65}
          rotationIntensity={0.05}
          floatIntensity={0.12}
        >
          <Scene />
        </Float>

        <OrbitControls
          enablePan={false}
          enableZoom={false}
          enableDamping
          dampingFactor={0.08}
          autoRotate
          autoRotateSpeed={0.08}
          minPolarAngle={
            Math.PI / 2.3
          }
          maxPolarAngle={
            Math.PI / 1.7
          }
        />

      </Canvas>

    </div>
  );
}