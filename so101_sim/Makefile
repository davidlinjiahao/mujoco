# Makefile for SO-101 MuJoCo Simulation

.PHONY: help install record-base record-grounded deploy-base deploy-grounded clean

help:
	@echo "SO-101 MuJoCo Simulation Makefile"
	@echo "Testing: Base Sim → Grounded Sim → Sim2Real"
	@echo ""
	@echo "Available targets:"
	@echo "  install          - Install all dependencies"
	@echo "  record-base      - Record demos in base simulation (simple shapes)"
	@echo "  record-grounded  - Record demos in grounded simulation (realistic scans)"
	@echo "  deploy-base      - Deploy policy in base simulation"
	@echo "  deploy-grounded  - Deploy policy in grounded simulation"
	@echo "  clean            - Remove temporary files"
	@echo ""
	@echo "Examples:"
	@echo "  make record-base PORT=/dev/tty.usbmodem58760431551"
	@echo "  make record-grounded PORT=/dev/tty.usbmodem58760431551"
	@echo "  make deploy-grounded POLICY=outputs/train/act/checkpoints/last/pretrained_model"
	@echo ""
	@echo "LeRobot Training (run directly):"
	@echo "  lerobot-train --dataset.repo_id=USER/so101_sim --policy.type=act"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Login to HuggingFace: huggingface-cli login"
	@echo "  2. Calibrate leader arm: lerobot-calibrate --robot.type=so101_leader ..."
	@echo "  3. Record demos: make record-base PORT=/dev/tty.usbmodem..."

record-base:
	@echo "🎮 Recording in BASE SIMULATION (simple primitives)"
	@echo "Web control panel: http://127.0.0.1:5001"
	@echo ""
	@if [ -z "$(PORT)" ]; then \
		echo "Error: PORT not specified"; \
		echo "Usage: make record-base PORT=/dev/tty.usbmodem58760431551 ID=my_leader_arm"; \
		exit 1; \
	fi
	python3 scripts/teleoperate.py \
		--scene scenes/table.xml \
		--scene scenes/cube.xml \
		--scene scenes/bin.xml \
		--leader-port $(PORT) \
		--leader-id $(or $(ID),my_leader_arm) \
		--render \
		--record

record-grounded:
	@echo "🏗️ Recording in GROUNDED SIMULATION (realistic scans)"
	@echo "Web control panel: http://127.0.0.1:5001"
	@echo ""
	@if [ -z "$(PORT)" ]; then \
		echo "Error: PORT not specified"; \
		echo "Usage: make record-grounded PORT=/dev/tty.usbmodem58760431551 ID=my_leader_arm"; \
		exit 1; \
	fi
	python3 scripts/teleoperate.py \
		--scene scenes/grounded_scene.xml \
		--leader-port $(PORT) \
		--leader-id $(or $(ID),my_leader_arm) \
		--render \
		--record

deploy-base:
	@echo "🤖 Deploying policy in BASE SIMULATION"
	@if [ -z "$(POLICY)" ]; then \
		echo "Error: POLICY path not specified"; \
		echo "Usage: make deploy-base POLICY=outputs/train/act/checkpoints/last/pretrained_model"; \
		exit 1; \
	fi
	python3 scripts/deploy.py \
		--scene scenes/table.xml \
		--scene scenes/cube.xml \
		--scene scenes/bin.xml \
		--policy-path $(POLICY) \
		--episodes $(or $(EPISODES),10) \
		--render \
		$(if $(RECORD),--record,) \
		--device $(or $(DEVICE),mps)

deploy-grounded:
	@echo "🏗️ Deploying policy in GROUNDED SIMULATION"
	@echo "Testing generalization to realistic scans..."
	@if [ -z "$(POLICY)" ]; then \
		echo "Error: POLICY path not specified"; \
		echo "Usage: make deploy-grounded POLICY=outputs/train/act/checkpoints/last/pretrained_model"; \
		exit 1; \
	fi
	python3 scripts/deploy.py \
		--scene scenes/grounded_scene.xml \
		--policy-path $(POLICY) \
		--episodes $(or $(EPISODES),10) \
		--render \
		$(if $(RECORD),--record,) \
		--device $(or $(DEVICE),mps)

clean:
	@echo "Cleaning temporary files..."
	rm -f scenes/_merged_*.xml
	rm -rf __pycache__
	rm -rf scripts/__pycache__
	rm -rf scripts/utilities/__pycache__
	rm -rf sims/__pycache__
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete
	@echo "✓ Cleanup complete"

.DEFAULT_GOAL := help
