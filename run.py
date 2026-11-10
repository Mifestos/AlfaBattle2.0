"""
Оркестратор пайплайна Alfa Battle 2.0.
"""
import importlib.util
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REQUIRED_MODULES = ["pandas", "numpy", "lightgbm", "sklearn", "torch", "catboost"]

class Color:
    CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"; RESET = "\033[0m"

STEPS = {
    "v1":       ("v01_baseline_gbdt.train",              "v01: LightGBM baseline"),
    "dl":       ("v02_deep_learning.train_dl",           "v02: Transformer OOF"),
    "catboost": ("v03_catboost.train_cb",                "v03: CatBoost OOF"),
    "blend":    ("v04_ensemble.blend",                   "v04: Final Blending (3-Way)"),
    "inference":("v04_ensemble.inference",               "v05: Generate Submission"),
}

MENU_OPTIONS = {
    "1": {"title": "Полный цикл (Train + Submit)", "desc": "Обучение всех моделей и генерация ответа", "steps": ["v1", "dl", "catboost", "blend", "inference"]},
    "2": {"title": "Только блендинг", "desc": "Смешать готовые OOF-предсказания", "steps": ["blend"]},
    "3": {"title": "Сгенерировать сабмит", "desc": "Требует готовые модели и веса", "steps": ["inference"]},
    "4": {"title": "Только baseline", "desc": "Только LightGBM", "steps": ["v1"]},
    "5": {"title": "Только Transformer", "desc": "Только DL", "steps": ["dl"]},
    "6": {"title": "Только CatBoost", "desc": "Только CatBoost", "steps": ["catboost"]},
}

def clear_screen(): os.system("cls" if os.name == "nt" else "clear")

def setup_logging():
    log_path = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout); console.setFormatter(fmt); logger.addHandler(console)
    file_handler = logging.FileHandler(log_path, encoding="utf-8"); file_handler.setFormatter(fmt); logger.addHandler(file_handler)
    return logger

def check_environment() -> tuple[bool, str]:
    missing = [mod for mod in REQUIRED_MODULES if importlib.util.find_spec(mod) is None]
    if missing: return False, f"Не установлены: {', '.join(missing)}"
    if not (PROJECT_ROOT / "archive" / "alfabattle2_train_target.csv").exists():
        return False, "Нет файлов в archive/"
    import torch
    gpu_info = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    return True, gpu_info

def show_menu(gpu_info: str) -> list[str] | None:
    clear_screen()
    print(f"{Color.CYAN}{Color.BOLD}" + "=" * 65)
    print("  ALFA BATTLE 2.0 - PIPELINE ORCHESTRATOR")
    print("=" * 65 + f"{Color.RESET}")
    print(f"  Среда: {Color.GREEN}{gpu_info}{Color.RESET}")
    print(f"  Проект: {PROJECT_ROOT}\n")
    print(f"{Color.BOLD}  Выберите режим запуска:{Color.RESET}")
    for key, opt in MENU_OPTIONS.items():
        rec = f" {Color.YELLOW}[рекомендуется]{Color.RESET}" if key == "1" else ""
        print(f"  [{Color.CYAN}{key}{Color.RESET}] {Color.BOLD}{opt['title']}{Color.RESET}{rec}")
        print(f"      {Color.GREEN}->{Color.RESET} {opt['desc']}")
        print(f"      {Color.YELLOW}Шаги:{Color.RESET} {' -> '.join(opt['steps'])}\n")
    print(f"  [{Color.CYAN}0{Color.RESET}] Выход")
    print(f"{Color.CYAN}" + "=" * 65 + f"{Color.RESET}")

    while True:
        try:
            choice = input(f"\n  {Color.BOLD}Введите номер > {Color.RESET}").strip()
            if choice == "0": return None
            if choice in MENU_OPTIONS: return MENU_OPTIONS[choice]["steps"]
            print(f"  {Color.RED}Ошибка: введите число от 0 до 6.{Color.RESET}")
        except KeyboardInterrupt: return None

def run_step(module: str, description: str, logger: logging.Logger) -> float:
    print(f"\n{Color.CYAN}[START]{Color.RESET} {description}")
    logger.info(f"Запуск: {module}")
    t0 = time.time()
    try:
        result = subprocess.run([sys.executable, "-m", module], cwd=PROJECT_ROOT, check=False)
    except KeyboardInterrupt:
        sys.exit(130)
    elapsed = (time.time() - t0) / 60
    if result.returncode != 0:
        print(f"\n{Color.RED}[FAIL]{Color.RESET} {description} (exit={result.returncode})")
        sys.exit(result.returncode)
    print(f"{Color.GREEN}[OK]{Color.RESET} {description} за {elapsed:.1f} мин")
    return elapsed

def main() -> None:
    ok, info = check_environment()
    if not ok:
        print(f"\n{Color.RED}[ERROR] {info}{Color.RESET}"); sys.exit(1)
    steps_to_run = show_menu(info)
    if not steps_to_run: sys.exit(0)

    logger = setup_logging()
    total_t0 = time.time()
    timings = {}

    for step in steps_to_run:
        module, description = STEPS[step]
        timings[step] = run_step(module, description, logger)

    total_time = (time.time() - total_t0) / 60
    print(f"\n{Color.CYAN}{Color.BOLD}" + "=" * 65)
    print("  ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО")
    print("=" * 65 + f"{Color.RESET}")
    for step, mins in timings.items():
        print(f"  {Color.GREEN}OK{Color.RESET} {STEPS[step][1]:<30} {mins:>6.1f} мин")
    print(f"  {Color.BOLD}Итого:{Color.RESET} {total_time:.1f} мин")
    print(f"{Color.CYAN}" + "=" * 65 + f"{Color.RESET}\n")

if __name__ == "__main__":
    main()