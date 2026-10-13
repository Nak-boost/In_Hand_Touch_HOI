"""Plot the checked-in official DIGIT surface and derived collision envelope."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pybullet as pb

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'assets/allegro'
PACKAGE = ASSETS / 'official_digit'


def main():
    manifest = json.loads((PACKAGE / 'mesh_manifest.json').read_text())
    fig = plt.figure(figsize=(12, 6))
    client = pb.connect(pb.DIRECT)
    try:
        for column, kind in enumerate(['visual', 'collision']):
            pb.resetSimulation(physicsClientId=client)
            ax = fig.add_subplot(1, 2, column+1)
            for part, color in [('digit_housing', [.47,.49,.52,1]), ('digit_gel', [.15,.75,.83,1])]:
                paths = ['official_digit/visual/'+part+'.obj'] if kind == 'visual' else manifest['collision'][part]
                for path in paths:
                    visual = pb.createVisualShape(pb.GEOM_MESH, fileName=str(ASSETS/path),
                                                  rgbaColor=color, physicsClientId=client)
                    pb.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, physicsClientId=client)
            # CPU depth-buffer rendering handles occlusion between the dense
            # casing and gel surfaces, without matplotlib painter artifacts.
            view = pb.computeViewMatrix([.09,-.065,.055], [.004,0,.018], [0,0,1])
            projection = pb.computeProjectionMatrixFOV(28, 1, .001, 1)
            pixels = pb.getCameraImage(800, 800, view, projection,
                                       renderer=pb.ER_TINY_RENDERER, physicsClientId=client)[2]
            ax.imshow(np.asarray(pixels).reshape(800,800,4))
            ax.axis('off')
            ax.set_title('Published TACTO surface' if kind == 'visual' else 'Rigid collision approximation')
    finally:
        pb.disconnect(client)
    fig.suptitle('DIGIT: original geometry, no size rescaling\nCyan: curved gel; grey: housing', fontsize=14)
    fig.tight_layout()
    output = ROOT.parent / 'research/digit_mount_review/digit_detail.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
