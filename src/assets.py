import customtkinter as ctk
from PIL import Image
from pathlib import Path

class Icons:
    hexagon=None
    check=None
    
    @classmethod
    def load_all(cls):
        base_dir=Path(__file__).parent.parent
        icons_dir=base_dir/"assets"/"icons"
        DEFAULT_SIZE=(16,16)
        
        def load_img(light_name, dark_name=None, size=DEFAULT_SIZE):
            if dark_name is None:
                dark_name=light_name
            return ctk.CTkImage(light_image=Image.open(icons_dir/light_name), dark_image=Image.open(icons_dir/dark_name), size=size)
        
        
        
        cls.hexagon=load_img("hexagon_light.png", "hexagon_dark.png")
        cls.check=load_img("check.png", "check.png")
        cls.hexagon_warning=load_img("hexagon_warning.png", "hexagon_warning.png")
        cls.stop=load_img("stop.png", "stop.png")
        cls.refresh=load_img("refresh.png", "refresh.png")
        cls.dataset=load_img("dataset.png", "dataset.png")
        cls.config=load_img("config.png", "config.png")
        cls.pipeline=load_img("pipeline.png", "pipeline.png")
        cls.run=load_img("run.png", "run.png")
        cls.results=load_img("results.png", "results.png")
        cls.redirect=load_img("redirect.png", "redirect.png")
        cls.circle=load_img("circle.png", "circle.png")
        cls.error=load_img("error.png", "error.png")
        cls.skip=load_img("skip.png", "skip.png")