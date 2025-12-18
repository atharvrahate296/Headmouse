"""
Virtual Mouse - Main GUI Application
Bridging the gap between Human Computer Interaction (HCI)
"""

import tkinter as tk
from tkinter import messagebox, ttk
import sys
import threading
from pathlib import Path
from PIL import Image, ImageTk

# Try importing the mouse control modules
try:
    from hand_mouse import handmouse
    HAND_MOUSE_AVAILABLE = True
except ImportError as e:
    HAND_MOUSE_AVAILABLE = False
    print(f"Warning: hand_mouse.py not available: {e}")

try:
    from head_mouse import headmouse
    HEAD_MOUSE_AVAILABLE = True
except ImportError as e:
    HEAD_MOUSE_AVAILABLE = False
    print(f"Warning: head_mouse.py not available: {e}")


class VirtualMouseGUI:
    """Main GUI application for Virtual Mouse control system"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Mouse - HCI Interface")
        self.root.geometry("1200x700+150+30")
        self.root.resizable(False, False)
        
        # Set color scheme
        self.bg_primary = "#252528"
        self.bg_secondary = "#2b2b30"
        self.accent_color = "#0f3460"
        self.highlight_color = "#bb6d7a"
        self.button_color = "#5be480"
        self.text_color = "#eaeaea"
        self.info_color = "#3b82f6"
        
        self.root.configure(bg=self.bg_primary)
        
        # Running state
        self.is_running = False
        
        self.setup_ui()
        
        # Center window on screen
        self.center_window()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def show_hand_mouse_info(self):
        """Display detailed Hand Mouse instructions"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Hand Mouse - Control Guide")
        info_window.geometry("650x600")
        info_window.configure(bg=self.bg_secondary)
        info_window.resizable(False, False)
        
        # Center the info window
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (650 // 2)
        y = (info_window.winfo_screenheight() // 2) - (600 // 2)
        info_window.geometry(f"650x600+{x}+{y}")
        
        # Title
        title = tk.Label(
            info_window,
            text="🖐️ Hand Mouse Control Guide",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_secondary,
            fg=self.highlight_color
        )
        title.pack(pady=(20, 10))
        
        # Scrollable frame
        canvas = tk.Canvas(info_window, bg=self.bg_secondary, highlightthickness=0)
        scrollbar = tk.Scrollbar(info_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_secondary)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Instructions content
        instructions = [
            ("Cursor Movement", [
                "• Open Hand (4-5 fingers extended) - Move cursor freely",
                "• Cursor tracks your wrist position",
                "• Smooth movement with dead zone for stability"
            ]),
            ("Click Controls", [
                "• Index Finger + Thumb Pinch - Left Click",
                "• Middle Finger + Thumb Pinch - Right Click",
                "• Pinch gestures must be held for confirmation"
            ]),
            ("Scrolling", [
                "• Peace Sign (✌️ Index + Middle extended) - Toggle Scroll Mode",
                "• When scroll mode is ON, move index finger up/down to scroll",
                "• Peace sign again to turn OFF scroll mode"
            ]),
            ("Cursor Lock", [
                "• Fist (0-1 fingers extended) - Lock/Unlock Cursor",
                "• Prevents accidental cursor movement",
                "• Fist again to unlock"
            ]),
            ("Volume Control", [
                "• Thumb + Pinky Tip Touch - Volume Up",
                "• Thumb + Pinky Bottom Touch - Volume Down",
                "• Quick touch at top or bottom of pinky finger"
            ]),
            ("Keyboard Shortcuts", [
                "• ESC or Q - Exit the application",
                "• Works on both PIP window and camera feed"
            ]),
            ("Tips for Best Performance", [
                "• Ensure good lighting for camera",
                "• Keep hand within camera frame",
                "• Hold gestures steady for 3 frames to confirm",
                "• PIP window allows multitasking while controlling"
            ])
        ]
        
        for section, items in instructions:
            # Section header
            header = tk.Label(
                scrollable_frame,
                text=section,
                font=("Segoe UI", 12, "bold"),
                bg=self.bg_secondary,
                fg=self.info_color,
                anchor="w"
            )
            header.pack(fill=tk.X, padx=30, pady=(15, 5))
            
            # Section items
            for item in items:
                item_label = tk.Label(
                    scrollable_frame,
                    text=item,
                    font=("Segoe UI", 10),
                    bg=self.bg_secondary,
                    fg=self.text_color,
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=570
                )
                item_label.pack(fill=tk.X, padx=40, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(
            info_window,
            text="Got It!",
            font=("Segoe UI", 11, "bold"),
            bg=self.button_color,
            fg="white",
            cursor="hand2",
            command=info_window.destroy,
            relief=tk.FLAT,
            padx=20,
            pady=6
        )
        close_btn.pack(pady=15)
        close_btn.place(relx=0.78,rely=0.88)
    
    def show_head_mouse_info(self):
        """Display detailed Head Mouse instructions"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Head Mouse - Control Guide")
        info_window.geometry("650x600")
        info_window.configure(bg=self.bg_secondary)
        info_window.resizable(False, False)
        
        # Center the info window
        info_window.update_idletasks()
        x = (info_window.winfo_screenwidth() // 2) - (650 // 2)
        y = (info_window.winfo_screenheight() // 2) - (600 // 2)
        info_window.geometry(f"650x600+{x}+{y}")
        
        # Title
        title = tk.Label(
            info_window,
            text="👤 Head Mouse Control Guide",
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_secondary,
            fg=self.highlight_color
        )
        title.pack(pady=(20, 10))
        
        # Scrollable frame
        canvas = tk.Canvas(info_window, bg=self.bg_secondary, highlightthickness=0)
        scrollbar = tk.Scrollbar(info_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.bg_secondary)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Instructions content
        instructions = [
            ("Movement Modes", [
                "• RELATIVE Mode (Default) - Nose velocity-based control",
                "  - Move nose from center for cursor movement",
                "  - Small movements = slow cursor, Large = fast cursor",
                "  - Auto-recalibrates center when mouth closes",
                "• ABSOLUTE Mode - Iris position-based control",
                "  - Direct eye tracking for cursor position",
                "  - Press 'R' to toggle between modes"
            ]),
            ("Cursor Control", [
                "• OPEN MOUTH - Activate cursor movement",
                "  - In RELATIVE: Move nose to control direction/speed",
                "  - In ABSOLUTE: Eye position controls cursor",
                "• CLOSE MOUTH - Lock cursor & recalibrate center",
                "  - Stabilizes cursor at current position",
                "  - Resets center point in relative mode"
            ]),
            ("Click Controls", [
                "• Wink Left Eye - Left Click",
                "• Wink Right Eye - Right Click",
                "• Clicks only work when mouth is CLOSED",
                "• Hold wink for 6 frames to confirm (prevents accidents)"
            ]),
            ("Scrolling", [
                "• Blink Both Eyes - Toggle Scroll Mode ON/OFF",
                "• Works only when mouth is CLOSED",
                "• Move head up/down to scroll when active",
                "• Blink again to disable scrolling"
            ]),
            ("Keyboard Shortcuts", [
                "• ESC - Exit the application",
                "• P - Toggle PIP (Picture-in-Picture) mode",
                "• R - Toggle between Relative/Absolute mode",
                "• D - Toggle debug values display",
                "• M - Toggle face mesh visualization",
                "• N - Toggle nose control circle (relative mode)",
                "• All shortcuts work on both windows"
            ]),
            ("Visual Indicators", [
                "• Yellow Circle - Control radius (relative mode)",
                "• Gray Inner Circle - Dead zone (no movement)",
                "• Green Dot - Reference center point",
                "• Cyan Dot - Current nose position",
                "• Speed Bar - Shows current movement speed",
                "• 'MOVING: ACTIVE' - Cursor is being controlled",
                "• 'CURSOR: LOCKED' - Cursor is stationary"
            ]),
            ("Tips for Best Performance", [
                "• Ensure good lighting on your face",
                "• Keep face centered in camera view",
                "• Use relative mode for precise control",
                "• Close mouth briefly to recalibrate center",
                "• Blink naturally - holds must be deliberate",
                "• PIP mode allows working while controlling"
            ])
        ]
        
        for section, items in instructions:
            # Section header
            header = tk.Label(
                scrollable_frame,
                text=section,
                font=("Segoe UI", 12, "bold"),
                bg=self.bg_secondary,
                fg=self.info_color,
                anchor="w"
            )
            header.pack(fill=tk.X, padx=30, pady=(15, 5))
            
            # Section items
            for item in items:
                item_label = tk.Label(
                    scrollable_frame,
                    text=item,
                    font=("Segoe UI", 10),
                    bg=self.bg_secondary,
                    fg=self.text_color,
                    anchor="w",
                    justify=tk.LEFT,
                    wraplength=570
                )
                item_label.pack(fill=tk.X, padx=40, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(
            info_window,
            text="Got It!",
            font=("Segoe UI", 11, "bold"),
            bg=self.button_color,
            fg="white",
            cursor="hand2",
            command=info_window.destroy,
            relief=tk.FLAT,
            padx=20,
            pady=6
        )
        close_btn.pack(pady=15)
        close_btn.place(relx=0.78,rely=0.88)
    
    def setup_ui(self):
        """Setup the main user interface"""
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_primary)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Header section
        header_frame = tk.Frame(main_frame, bg=self.bg_primary)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="Virtual Mouse",
            font=("Segoe UI", 42, "bold"),
            bg=self.bg_primary,
            fg=self.highlight_color
        )
        title_label.pack()
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Bridging the gap between Human Computer Interaction (HCI)",
            font=("Segoe UI", 14, "italic"),
            bg=self.bg_primary,
            fg=self.text_color
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Divider
        divider = tk.Frame(header_frame, bg=self.accent_color, height=2)
        divider.pack(fill=tk.X, pady=(20, 0))
        
        # Content section
        content_frame = tk.Frame(main_frame, bg=self.bg_primary)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        # Description
        desc_frame = tk.Frame(content_frame, bg=self.bg_secondary, bd=0)
        desc_frame.pack(fill=tk.X, pady=(0, 30))
        
        desc_text = (
            "Choose your preferred control method:\n\n"
            "• Hand Mouse - Control your cursor using hand gestures\n"
            "• Head Mouse - Control your cursor using head and facial movements"
        )
        
        desc_label = tk.Label(
            desc_frame,
            text=desc_text,
            font=("Segoe UI", 11),
            bg=self.bg_secondary,
            fg=self.text_color,
            justify=tk.LEFT,
            padx=30,
            pady=20
        )
        desc_label.pack(fill=tk.X)
        
        # Buttons container
        buttons_frame = tk.Frame(content_frame, bg=self.bg_primary)
        buttons_frame.pack(expand=True)
        
        # Configure grid
        buttons_frame.grid_columnconfigure(0, weight=1)
        buttons_frame.grid_columnconfigure(1, weight=1)
        
        # Hand Mouse Button
        self.hand_btn = self.create_mode_button(
            buttons_frame,
            "🖐️ Hand Mouse",
            "Gesture-based control using\nyour hand movements",
            self.launch_hand_mouse,
            self.show_hand_mouse_info,
            0,
            HAND_MOUSE_AVAILABLE
        )
        
        # Head Mouse Button
        self.head_btn = self.create_mode_button(
            buttons_frame,
            "👤 Head Mouse",
            "Control using facial\nand head movements",
            self.launch_head_mouse,
            self.show_head_mouse_info,
            1,
            HEAD_MOUSE_AVAILABLE
        )
        
        # Footer
        footer_frame = tk.Frame(main_frame, bg=self.bg_primary)
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        footer_text = "Press ESC or 'Q' in the camera window to exit • Ensure your camera is connected"
        footer_label = tk.Label(
            footer_frame,
            text=footer_text,
            font=("Segoe UI", 9),
            bg=self.bg_primary,
            fg=self.text_color,
            justify=tk.CENTER
        )
        footer_label.pack()
        
        # Status indicator
        self.status_label = tk.Label(
            footer_frame,
            text="● Ready",
            font=("Segoe UI", 10, "bold"),
            bg=self.bg_primary,
            fg="#4ade80"
        )
        self.status_label.pack(pady=(10, 0))
    
    def create_mode_button(self, parent, title, description, command, info_command, column, enabled):
        """Create a styled mode selection button with info icon"""
        
        # Button container
        btn_container = tk.Frame(
            parent,
            bg=self.bg_secondary,
            highlightbackground=self.accent_color,
            highlightthickness=2
        )
        btn_container.grid(row=0, column=column, padx=20, pady=10, sticky="nsew")
        
        # Make button container clickable
        if enabled:
            btn_container.bind("<Button-1>", lambda e: command())
            btn_container.bind("<Enter>", lambda e: self.on_hover_enter(btn_container))
            btn_container.bind("<Leave>", lambda e: self.on_hover_leave(btn_container))
        else:
            btn_container.configure(bg="#2a2a3e", highlightbackground="#3a3a4e")
        
        # Content frame
        content = tk.Frame(btn_container, bg=self.bg_secondary)
        content.pack(padx=30, pady=30)
        
        # Title with info button container
        title_container = tk.Frame(content, bg=self.bg_secondary)
        title_container.pack(pady=(0, 10))
        
        # Title
        title_label = tk.Label(
            title_container,
            text=title,
            font=("Segoe UI", 18, "bold"),
            bg=self.bg_secondary,
            fg=self.text_color if enabled else "#666666"
        )
        title_label.pack(side=tk.LEFT, padx=(0, 10))
        
        if enabled:
            title_label.bind("<Button-1>", lambda e: command())
            title_label.bind("<Enter>", lambda e: self.on_hover_enter(btn_container))
            title_label.bind("<Leave>", lambda e: self.on_hover_leave(btn_container))
            
            # Info button
            info_btn = tk.Label(
                title_container,
                text="ℹ️",
                font=("Segoe UI", 16),
                bg=self.bg_secondary,
                fg=self.info_color,
                cursor="hand2"
            )
            info_btn.pack(side=tk.LEFT)
            info_btn.bind("<Button-1>", lambda e: info_command())
            info_btn.bind("<Enter>", lambda e: info_btn.config(fg=self.highlight_color))
            info_btn.bind("<Leave>", lambda e: info_btn.config(fg=self.info_color))
        
        # Description
        desc_label = tk.Label(
            content,
            text=description,
            font=("Segoe UI", 10),
            bg=self.bg_secondary,
            fg=self.text_color if enabled else "#666666",
            justify=tk.CENTER
        )
        desc_label.pack()
        
        if enabled:
            desc_label.bind("<Button-1>", lambda e: command())
            desc_label.bind("<Enter>", lambda e: self.on_hover_enter(btn_container))
            desc_label.bind("<Leave>", lambda e: self.on_hover_leave(btn_container))
        
        # Status indicator
        if not enabled:
            status = tk.Label(
                content,
                text="⚠️ Not Available",
                font=("Segoe UI", 9),
                bg=self.bg_secondary,
                fg="#ef4444"
            )
            status.pack(pady=(10, 0))
        
        return btn_container
    
    def on_hover_enter(self, widget):
        """Handle mouse hover enter"""
        widget.configure(highlightbackground=self.highlight_color, highlightthickness=3)
    
    def on_hover_leave(self, widget):
        """Handle mouse hover leave"""
        widget.configure(highlightbackground=self.accent_color, highlightthickness=2)
    
    def update_status(self, message, color="#4ade80"):
        """Update status label"""
        self.status_label.config(text=f"● {message}", fg=color)
        self.root.update()
    
    def launch_hand_mouse(self):
        """Launch hand mouse control"""
        if not HAND_MOUSE_AVAILABLE:
            messagebox.showerror(
                "Error",
                "Hand Mouse module is not available.\n\n"
                "Please ensure hand_mouse.py is in the same directory."
            )
            return
        
        if self.is_running:
            messagebox.showwarning(
                "Already Running",
                "A mouse control session is already active.\n"
                "Please close it before starting another."
            )
            return
        
        self.is_running = True
        self.update_status("Hand Mouse Active", "#fbbf24")
        
        # Run in separate thread to avoid blocking GUI
        def run():
            try:
                handmouse()
            except Exception as e:
                messagebox.showerror("Error", f"Hand Mouse encountered an error:\n{e}")
            finally:
                self.is_running = False
                self.update_status("Ready", "#4ade80")
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    
    def launch_head_mouse(self):
        """Launch head mouse control"""
        if not HEAD_MOUSE_AVAILABLE:
            messagebox.showerror(
                "Error",
                "Head Mouse module is not available.\n\n"
                "Please ensure head_mouse.py is in the same directory."
            )
            return
        
        if self.is_running:
            messagebox.showwarning(
                "Already Running",
                "A mouse control session is already active.\n"
                "Please close it before starting another."
            )
            return
        
        self.is_running = True
        self.update_status("Head Mouse Active", "#fbbf24")
        
        # Run in separate thread to avoid blocking GUI
        def run():
            try:
                headmouse()
            except Exception as e:
                messagebox.showerror("Error", f"Head Mouse encountered an error:\n{e}")
            finally:
                self.is_running = False
                self.update_status("Ready", "#4ade80")
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = VirtualMouseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()