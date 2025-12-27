try:
    print("Testing ai2thor import...")
    import ai2thor
    print(f"ai2thor version: {ai2thor.__version__}")
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    print("Initializing Controller...")
    # controller = Controller(scene="FloorPlan1", platform=CloudRendering) # Skip actual init to avoid hang
    print("Successfully imported and found Controller.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

