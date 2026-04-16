import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw

THUMB_SIZE = 250

class EditWindow(tk.Toplevel):
    def __init__(self, master, image_path, label_path, main_app):
        super().__init__(master)
        self.title(f"Editing - {os.path.basename(image_path)}")
        self.main_app = main_app
        self.image_path = image_path
        self.label_path = label_path
        
        self.bboxes = []
        self.selected_bbox_idx = -1
        self.scale_factor = 1.0
        
        self.setup_ui()
        self.load_image()
        self.load_labels()
        self.draw_bboxes()
        
    def setup_ui(self):
        self.canvas = tk.Canvas(self, cursor="cross", bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<ButtonPress-3>", self.select_bbox)
        self.bind("<Delete>", self.delete_bbox)
        self.bind("<s>", lambda e: self.save_labels())
        
        self.rect = None
        self.start_x = None; self.start_y = None
        self.cur_x = None; self.cur_y = None
        
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        tk.Button(btn_frame, text="Save (s)", command=self.save_labels).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Delete BBox (Del)", command=lambda: self.delete_bbox(None)).pack(side=tk.LEFT, padx=2)
        
        tk.Label(btn_frame, text="Class ID:").pack(side=tk.LEFT, padx=2)
        self.class_id_var = tk.StringVar(value="0")
        tk.Entry(btn_frame, textvariable=self.class_id_var, width=5).pack(side=tk.LEFT, padx=2)
        
        tk.Label(btn_frame, text="Right-click to select. Drag to draw.").pack(side=tk.LEFT, padx=10)

    def load_image(self):
        self.current_image = Image.open(self.image_path)
        self.img_width, self.img_height = self.current_image.size
        
        # Max dimensions for single editing screen
        max_w, max_h = 1000, 700
        self.scale_factor = min(1.0, max_w / self.img_width, max_h / self.img_height)
        
        if self.scale_factor < 1.0:
            new_w = int(self.img_width * self.scale_factor)
            new_h = int(self.img_height * self.scale_factor)
            resized = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(resized)
        else:
            self.tk_image = ImageTk.PhotoImage(self.current_image)
            
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def load_labels(self):
        if os.path.exists(self.label_path):
            with open(self.label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_c, y_c = float(parts[1]), float(parts[2])
                        w, h = float(parts[3]), float(parts[4])
                        self.bboxes.append([class_id, x_c, y_c, w, h])

    def save_labels(self):
        if not self.bboxes:
            if os.path.exists(self.label_path):
                os.remove(self.label_path)
        else:
            os.makedirs(os.path.dirname(self.label_path), exist_ok=True)
            with open(self.label_path, 'w') as f:
                for bbox in self.bboxes:
                    f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
                    
        self.main_app.refresh_gallery(self.image_path)
        self.title(f"Editing - {os.path.basename(self.image_path)} [SAVED]")

    def draw_bboxes(self):
        self.canvas.delete("bbox")
        for idx, bbox in enumerate(self.bboxes):
            class_id, x_c, y_c, w, h = bbox
            x1 = (x_c - w/2) * self.img_width * self.scale_factor
            y1 = (y_c - h/2) * self.img_height * self.scale_factor
            x2 = (x_c + w/2) * self.img_width * self.scale_factor
            y2 = (y_c + h/2) * self.img_height * self.scale_factor
            
            color = "red" if idx == self.selected_bbox_idx else "lime"
            width = 3 if idx == self.selected_bbox_idx else 2
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags="bbox")
            self.canvas.create_text(x1, y1-10, text=str(class_id), fill=color, font=("Arial", 12, "bold"), tags="bbox", anchor=tk.SW)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='blue', width=2)
        self.selected_bbox_idx = -1
        self.draw_bboxes()

    def on_move_press(self, event):
        self.cur_x, self.cur_y = event.x, event.y
        c_w = self.img_width * self.scale_factor
        c_h = self.img_height * self.scale_factor
        self.cur_x = max(0, min(c_w, self.cur_x))
        self.cur_y = max(0, min(c_h, self.cur_y))
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, self.cur_x, self.cur_y) # type: ignore

    def on_button_release(self, event):
        if not self.rect: return
        x1, x2 = sorted([self.start_x, self.cur_x if self.cur_x else self.start_x]) # type: ignore
        y1, y2 = sorted([self.start_y, self.cur_y if self.cur_y else self.start_y]) # type: ignore
        if x2 - x1 > 5 and y2 - y1 > 5:
            try: class_id = int(self.class_id_var.get())
            except: class_id = 0; self.class_id_var.set("0")
            
            orig_x1 = x1 / self.scale_factor
            orig_x2 = x2 / self.scale_factor
            orig_y1 = y1 / self.scale_factor
            orig_y2 = y2 / self.scale_factor
            
            x_c = ((orig_x1 + orig_x2) / 2) / self.img_width
            y_c = ((orig_y1 + orig_y2) / 2) / self.img_height
            w = (orig_x2 - orig_x1) / self.img_width
            h = (orig_y2 - orig_y1) / self.img_height
            
            self.bboxes.append([class_id, x_c, y_c, w, h])
            self.selected_bbox_idx = len(self.bboxes) - 1
            
        self.canvas.delete(self.rect)
        self.rect = None
        self.draw_bboxes()

    def select_bbox(self, event):
        ex, ey = event.x, event.y
        self.selected_bbox_idx = -1
        for idx, bbox in enumerate(self.bboxes):
            class_id, x_c, y_c, w, h = bbox
            x1 = (x_c - w/2) * self.img_width * self.scale_factor
            y1 = (y_c - h/2) * self.img_height * self.scale_factor
            x2 = (x_c + w/2) * self.img_width * self.scale_factor
            y2 = (y_c + h/2) * self.img_height * self.scale_factor
            if x1 <= ex <= x2 and y1 <= ey <= y2:
                self.selected_bbox_idx = idx
                self.class_id_var.set(str(class_id))
                break
        self.draw_bboxes()

    def delete_bbox(self, event):
        if self.selected_bbox_idx != -1:
            del self.bboxes[self.selected_bbox_idx]
            self.selected_bbox_idx = -1
            self.draw_bboxes()

class GalleryApp:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLO Dataset Gallery")
        
        self.cols = 4
        # Calculate width to exactly fit 4 columns
        # Each column takes THUMB_SIZE + 20 (padx=10 left and right)
        # Add 40 for scrollbar and extra padding
        calc_width = self.cols * (THUMB_SIZE + 20) + 40
        self.master.geometry(f"{calc_width}x800")
        
        self.image_dir = ""
        self.label_dir = ""
        self.image_files = []
        self.thumbnails = {}
        
        self.page_size = 200
        self.current_page = 0
        self.current_job_id = 0
        
        top_frame = tk.Frame(master)
        top_frame.pack(fill=tk.X, side=tk.TOP, pady=5)
        tk.Button(top_frame, text="Load Dir", command=self.load_dir).pack(side=tk.LEFT, padx=10)
        
        self.btn_prev = tk.Button(top_frame, text="< Prev", command=self.prev_page, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        self.lbl_page = tk.Label(top_frame, text="Page: 0/0")
        self.lbl_page.pack(side=tk.LEFT, padx=5)
        self.btn_next = tk.Button(top_frame, text="Next >", command=self.next_page, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        self.status_lbl = tk.Label(top_frame, text="Current gallery: None")
        self.status_lbl.pack(side=tk.LEFT, padx=10)
        
        # Scrollable canvas setup
        self.canvas = tk.Canvas(master, bg="#202020")
        self.scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#202020")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mouse wheel scanning
        self.master.bind_all("<MouseWheel>", self.on_mousewheel)
        self.master.bind_all("<Button-4>", self.on_mousewheel) # Linux wheel up
        self.master.bind_all("<Button-5>", self.on_mousewheel) # Linux wheel down
        
        self.master.bind("<Configure>", self.on_resize)
        self.last_w = calc_width
        
        # Load default yolo_dataset if exists
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_dataset", "images")
        if os.path.exists(default_dir):
            self.master.after(100, lambda: self.load_dir(default_dir))

    def on_mousewheel(self, event):
        # Ignore scroll events in EditWindow to prevent gallery scrolling while editing
        if not isinstance(event.widget.winfo_toplevel(), type(self.master)):
            return
            
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def on_resize(self, event):
        if event.widget == self.master:
            w = self.master.winfo_width()
            if abs(w - self.last_w) > 50:
                # Subtract 40px to account for the scrollbar and some padding
                new_cols = max(1, (w - 40) // (THUMB_SIZE + 20))
                if new_cols != self.cols and self.image_files:
                    self.cols = new_cols
                    self.last_w = w
                    self.populate_gallery()

    def load_dir(self, force_dir=None):
        if force_dir:
            self.image_dir = force_dir
        else:
            self.image_dir = filedialog.askdirectory(title="Select Image Directory")
            
        if not self.image_dir: return
        parent = os.path.dirname(self.image_dir.rstrip('/\\'))
        base = os.path.basename(self.image_dir.rstrip('/\\'))
        if base == 'images':
            gl = os.path.join(parent, 'labels')
            self.label_dir = gl if os.path.exists(gl) else self.image_dir
        else:
            self.label_dir = self.image_dir
            
        valid_exts =('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_exts)]
        self.image_files.sort()
        
        self.current_page = 0
        self.populate_gallery()

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.populate_gallery()

    def next_page(self):
        max_page = max(0, len(self.image_files) - 1) // self.page_size
        if self.current_page < max_page:
            self.current_page += 1
            self.populate_gallery()

    def get_labels(self, base_name):
        lbl_path = os.path.join(self.label_dir, base_name + ".txt")
        bboxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bboxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        return bboxes

    def make_thumbnail(self, img_path, bboxes):
        img = Image.open(img_path)
        w, h = img.size
        # Draw bboxes exactly on thumb directly
        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            x_c, y_c, bw, bh = bbox[1], bbox[2], bbox[3], bbox[4]
            x1 = (x_c - bw/2) * w
            y1 = (y_c - bh/2) * h
            x2 = (x_c + bw/2) * w
            y2 = (y_c + bh/2) * h
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=int(max(2, w*0.005)))
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        return ImageTk.PhotoImage(img)

    def populate_gallery(self):
        self.current_job_id += 1
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.thumbnails.clear()
        self.image_labels = {}
        
        total_imgs = len(self.image_files)
        if total_imgs == 0:
            self.lbl_page.config(text="Page: 0/0")
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)
            return
            
        max_page = (total_imgs - 1) // self.page_size
        self.lbl_page.config(text=f"Page: {self.current_page + 1}/{max_page + 1}")
        self.btn_prev.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_page < max_page else tk.DISABLED)
        
        self.start_idx = self.current_page * self.page_size
        self.end_idx = min(self.start_idx + self.page_size, total_imgs)
        self.current_load_idx = self.start_idx
        
        self.status_lbl.config(text=f"Loading {self.start_idx + 1} to {self.end_idx} of {total_imgs} images...")
        self.master.after(10, lambda jid=self.current_job_id: self.load_next_thumbnail_chunk(jid))
        
    def load_next_thumbnail_chunk(self, job_id):
        if job_id != self.current_job_id: return
        
        if self.current_load_idx >= self.end_idx:
            self.status_lbl.config(text=f"Showing images {self.start_idx + 1} - {self.end_idx} out of {len(self.image_files)}")
            return
            
        chunk_size = 10
        chunk_end = min(self.current_load_idx + chunk_size, self.end_idx)

        for idx in range(self.current_load_idx, chunk_end):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(self.label_dir, base_name + ".txt")
            
            bboxes = self.get_labels(base_name)
            thumb = self.make_thumbnail(img_path, bboxes)
            self.thumbnails[img_path] = thumb 
            
            grid_idx = idx - self.start_idx 
            row = grid_idx // self.cols
            col = grid_idx % self.cols
            
            frame = tk.Frame(self.scrollable_frame, bg="black", bd=2, relief=tk.RIDGE)
            frame.grid(row=row, column=col, padx=10, pady=10)
            
            lbl_img = tk.Label(frame, image=thumb, bg="black", cursor="hand2")
            lbl_img.pack()
            lbl_txt = tk.Label(frame, text=img_name, bg="black", fg="white", cursor="hand2")
            lbl_txt.pack(fill=tk.X)
            
            # Store reference to image label for partial updates
            self.image_labels[img_path] = lbl_img
            
            def on_click(e, p=img_path, lp=lbl_path):
                EditWindow(self.master, p, lp, self)
                
            lbl_img.bind("<Button-1>", on_click)
            lbl_txt.bind("<Button-1>", on_click)

        self.current_load_idx = chunk_end
        self.status_lbl.config(text=f"Loading {self.current_load_idx} / {len(self.image_files)}")
        self.master.after(20, lambda jid=job_id: self.load_next_thumbnail_chunk(jid))

    def refresh_gallery(self, updated_img_path=None):
        if updated_img_path and hasattr(self, 'image_labels') and updated_img_path in self.image_labels:
            # Recreate just this one thumbnail
            base_name = os.path.splitext(os.path.basename(updated_img_path))[0]
            bboxes = self.get_labels(base_name)
            thumb = self.make_thumbnail(updated_img_path, bboxes)
            self.thumbnails[updated_img_path] = thumb
            self.image_labels[updated_img_path].configure(image=thumb)
        else:
            # Full repopulate
            self.populate_gallery()

if __name__ == "__main__":
    root = tk.Tk()
    app = GalleryApp(root)
    root.mainloop()
