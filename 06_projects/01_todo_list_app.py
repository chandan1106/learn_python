"""
Todo List Application

A simple command-line todo list application that demonstrates:
- Object-oriented programming
- File I/O for data persistence
- Command-line interface
- Error handling
- Data structures (lists, dictionaries)
"""

import os
import json
import datetime
from enum import Enum


class Priority(Enum):
    """Enum for task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class Task:
    """Class representing a single task in the todo list."""
    
    def __init__(self, title, description="", due_date=None, priority=Priority.MEDIUM, completed=False):
        """Initialize a new task."""
        self.title = title
        self.description = description
        self.created_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.due_date = due_date
        self.priority = priority
        self.completed = completed
    
    def mark_completed(self):
        """Mark the task as completed."""
        self.completed = True
    
    def mark_pending(self):
        """Mark the task as pending (not completed)."""
        self.completed = False
    
    def update_title(self, title):
        """Update the task title."""
        self.title = title
    
    def update_description(self, description):
        """Update the task description."""
        self.description = description
    
    def update_due_date(self, due_date):
        """Update the task due date."""
        self.due_date = due_date
    
    def update_priority(self, priority):
        """Update the task priority."""
        self.priority = priority
    
    def to_dict(self):
        """Convert the task to a dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "created_date": self.created_date,
            "due_date": self.due_date,
            "priority": self.priority.name,
            "completed": self.completed
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a task from a dictionary."""
        task = cls(
            title=data["title"],
            description=data.get("description", ""),
            due_date=data.get("due_date"),
            priority=Priority[data.get("priority", "MEDIUM")],
            completed=data.get("completed", False)
        )
        task.created_date = data.get("created_date", task.created_date)
        return task
    
    def __str__(self):
        """Return a string representation of the task."""
        status = "✓" if self.completed else "✗"
        priority_str = {
            Priority.LOW: "Low",
            Priority.MEDIUM: "Medium",
            Priority.HIGH: "High"
        }[self.priority]
        
        due_str = f", Due: {self.due_date}" if self.due_date else ""
        return f"[{status}] {self.title} (Priority: {priority_str}{due_str})"


class TodoList:
    """Class representing a todo list that contains multiple tasks."""
    
    def __init__(self, name="My Todo List"):
        """Initialize a new todo list."""
        self.name = name
        self.tasks = []
        self.file_path = f"{name.lower().replace(' ', '_')}.json"
    
    def add_task(self, task):
        """Add a task to the todo list."""
        self.tasks.append(task)
    
    def remove_task(self, index):
        """Remove a task from the todo list by index."""
        if 0 <= index < len(self.tasks):
            return self.tasks.pop(index)
        return None
    
    def get_task(self, index):
        """Get a task by index."""
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None
    
    def list_tasks(self, show_completed=True):
        """List all tasks, optionally filtering out completed tasks."""
        if not show_completed:
            return [task for task in self.tasks if not task.completed]
        return self.tasks
    
    def list_by_priority(self, priority):
        """List tasks filtered by priority."""
        return [task for task in self.tasks if task.priority == priority]
    
    def save_to_file(self):
        """Save the todo list to a JSON file."""
        data = {
            "name": self.name,
            "tasks": [task.to_dict() for task in self.tasks]
        }
        
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)
    
    @classmethod
    def load_from_file(cls, file_path):
        """Load a todo list from a JSON file."""
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        todo_list = cls(name=data.get("name", "My Todo List"))
        todo_list.file_path = file_path
        
        for task_data in data.get("tasks", []):
            todo_list.add_task(Task.from_dict(task_data))
        
        return todo_list
    
    def __str__(self):
        """Return a string representation of the todo list."""
        return f"{self.name} ({len(self.tasks)} tasks)"


class TodoApp:
    """Class representing the todo list application."""
    
    def __init__(self):
        """Initialize the todo list application."""
        self.todo_lists = {}
        self.current_list = None
    
    def create_list(self, name):
        """Create a new todo list."""
        if name in self.todo_lists:
            print(f"A list named '{name}' already exists.")
            return False
        
        self.todo_lists[name] = TodoList(name)
        self.current_list = name
        return True
    
    def load_list(self, file_path):
        """Load a todo list from a file."""
        todo_list = TodoList.load_from_file(file_path)
        if todo_list:
            self.todo_lists[todo_list.name] = todo_list
            self.current_list = todo_list.name
            return True
        return False
    
    def save_current_list(self):
        """Save the current todo list to a file."""
        if self.current_list:
            self.todo_lists[self.current_list].save_to_file()
            return True
        return False
    
    def switch_list(self, name):
        """Switch to a different todo list."""
        if name in self.todo_lists:
            self.current_list = name
            return True
        return False
    
    def get_current_list(self):
        """Get the current todo list."""
        if self.current_list:
            return self.todo_lists[self.current_list]
        return None
    
    def run(self):
        """Run the todo list application."""
        print("Welcome to the Todo List Application!")
        
        while True:
            self._print_menu()
            choice = input("Enter your choice: ").strip()
            
            if choice == "1":
                self._create_new_list()
            elif choice == "2":
                self._load_list()
            elif choice == "3":
                self._switch_list()
            elif choice == "4":
                self._add_task()
            elif choice == "5":
                self._list_tasks()
            elif choice == "6":
                self._mark_task_completed()
            elif choice == "7":
                self._edit_task()
            elif choice == "8":
                self._remove_task()
            elif choice == "9":
                self._filter_by_priority()
            elif choice == "10":
                self._save_list()
            elif choice == "0":
                print("Thank you for using the Todo List Application!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _print_menu(self):
        """Print the application menu."""
        current = f" ({self.current_list})" if self.current_list else ""
        print("\n" + "=" * 50)
        print(f"TODO LIST APPLICATION{current}")
        print("=" * 50)
        print("1. Create a new list")
        print("2. Load a list from file")
        print("3. Switch to another list")
        print("4. Add a new task")
        print("5. List all tasks")
        print("6. Mark task as completed")
        print("7. Edit a task")
        print("8. Remove a task")
        print("9. Filter tasks by priority")
        print("10. Save current list")
        print("0. Exit")
        print("=" * 50)
    
    def _create_new_list(self):
        """Create a new todo list."""
        name = input("Enter the name for the new list: ").strip()
        if name:
            if self.create_list(name):
                print(f"Created new list: {name}")
            else:
                print("Failed to create list.")
        else:
            print("List name cannot be empty.")
    
    def _load_list(self):
        """Load a todo list from a file."""
        file_path = input("Enter the file path: ").strip()
        if file_path:
            if self.load_list(file_path):
                print(f"Loaded list from {file_path}")
            else:
                print(f"Failed to load list from {file_path}")
        else:
            print("File path cannot be empty.")
    
    def _switch_list(self):
        """Switch to a different todo list."""
        if not self.todo_lists:
            print("No lists available. Create or load a list first.")
            return
        
        print("\nAvailable lists:")
        for i, name in enumerate(self.todo_lists.keys(), 1):
            print(f"{i}. {name}")
        
        try:
            choice = int(input("Enter the number of the list to switch to: "))
            if 1 <= choice <= len(self.todo_lists):
                name = list(self.todo_lists.keys())[choice - 1]
                self.switch_list(name)
                print(f"Switched to list: {name}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    def _add_task(self):
        """Add a new task to the current list."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        title = input("Enter task title: ").strip()
        if not title:
            print("Task title cannot be empty.")
            return
        
        description = input("Enter task description (optional): ").strip()
        
        due_date = input("Enter due date (YYYY-MM-DD) (optional): ").strip()
        if due_date and not self._validate_date(due_date):
            print("Invalid date format. Using no due date.")
            due_date = None
        
        print("Priority levels:")
        print("1. Low")
        print("2. Medium")
        print("3. High")
        priority_choice = input("Choose priority (1-3) [default: 2]: ").strip()
        
        priority = Priority.MEDIUM
        if priority_choice == "1":
            priority = Priority.LOW
        elif priority_choice == "3":
            priority = Priority.HIGH
        
        task = Task(title, description, due_date, priority)
        self.todo_lists[self.current_list].add_task(task)
        print(f"Added task: {task}")
    
    def _list_tasks(self):
        """List all tasks in the current list."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        todo_list = self.todo_lists[self.current_list]
        show_completed = input("Show completed tasks? (y/n) [default: y]: ").strip().lower() != "n"
        
        tasks = todo_list.list_tasks(show_completed)
        if not tasks:
            print("No tasks found.")
            return
        
        print(f"\nTasks in {todo_list.name}:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
    
    def _mark_task_completed(self):
        """Mark a task as completed."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        todo_list = self.todo_lists[self.current_list]
        tasks = todo_list.list_tasks()
        
        if not tasks:
            print("No tasks found.")
            return
        
        print("\nTasks:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
        
        try:
            choice = int(input("Enter the number of the task to mark as completed: "))
            if 1 <= choice <= len(tasks):
                task = tasks[choice - 1]
                task.mark_completed()
                print(f"Marked task as completed: {task}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    def _edit_task(self):
        """Edit a task in the current list."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        todo_list = self.todo_lists[self.current_list]
        tasks = todo_list.list_tasks()
        
        if not tasks:
            print("No tasks found.")
            return
        
        print("\nTasks:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
        
        try:
            choice = int(input("Enter the number of the task to edit: "))
            if 1 <= choice <= len(tasks):
                task = tasks[choice - 1]
                
                print("\nEdit Task:")
                print("1. Title")
                print("2. Description")
                print("3. Due Date")
                print("4. Priority")
                print("5. Toggle Completion Status")
                
                edit_choice = input("Enter your choice: ").strip()
                
                if edit_choice == "1":
                    title = input(f"Enter new title (current: {task.title}): ").strip()
                    if title:
                        task.update_title(title)
                        print(f"Updated title: {task}")
                
                elif edit_choice == "2":
                    description = input(f"Enter new description (current: {task.description}): ").strip()
                    task.update_description(description)
                    print(f"Updated description for: {task}")
                
                elif edit_choice == "3":
                    due_date = input(f"Enter new due date (YYYY-MM-DD) (current: {task.due_date}): ").strip()
                    if due_date and not self._validate_date(due_date):
                        print("Invalid date format. Not updating due date.")
                    else:
                        task.update_due_date(due_date if due_date else None)
                        print(f"Updated due date for: {task}")
                
                elif edit_choice == "4":
                    print("Priority levels:")
                    print("1. Low")
                    print("2. Medium")
                    print("3. High")
                    priority_choice = input("Choose new priority (1-3): ").strip()
                    
                    if priority_choice == "1":
                        task.update_priority(Priority.LOW)
                    elif priority_choice == "2":
                        task.update_priority(Priority.MEDIUM)
                    elif priority_choice == "3":
                        task.update_priority(Priority.HIGH)
                    else:
                        print("Invalid choice. Not updating priority.")
                        return
                    
                    print(f"Updated priority for: {task}")
                
                elif edit_choice == "5":
                    if task.completed:
                        task.mark_pending()
                        print(f"Marked task as pending: {task}")
                    else:
                        task.mark_completed()
                        print(f"Marked task as completed: {task}")
                
                else:
                    print("Invalid choice.")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    def _remove_task(self):
        """Remove a task from the current list."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        todo_list = self.todo_lists[self.current_list]
        tasks = todo_list.list_tasks()
        
        if not tasks:
            print("No tasks found.")
            return
        
        print("\nTasks:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
        
        try:
            choice = int(input("Enter the number of the task to remove: "))
            if 1 <= choice <= len(tasks):
                task = todo_list.remove_task(choice - 1)
                print(f"Removed task: {task}")
            else:
                print("Invalid choice.")
        except ValueError:
            print("Please enter a valid number.")
    
    def _filter_by_priority(self):
        """Filter tasks by priority."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        todo_list = self.todo_lists[self.current_list]
        
        print("Priority levels:")
        print("1. Low")
        print("2. Medium")
        print("3. High")
        
        priority_choice = input("Choose priority to filter by (1-3): ").strip()
        
        priority = None
        if priority_choice == "1":
            priority = Priority.LOW
        elif priority_choice == "2":
            priority = Priority.MEDIUM
        elif priority_choice == "3":
            priority = Priority.HIGH
        else:
            print("Invalid choice.")
            return
        
        tasks = todo_list.list_by_priority(priority)
        
        if not tasks:
            print(f"No tasks found with {priority.name} priority.")
            return
        
        print(f"\nTasks with {priority.name} priority:")
        for i, task in enumerate(tasks, 1):
            print(f"{i}. {task}")
    
    def _save_list(self):
        """Save the current list to a file."""
        if not self.current_list:
            print("No active list. Create or load a list first.")
            return
        
        if self.save_current_list():
            todo_list = self.todo_lists[self.current_list]
            print(f"Saved list to {todo_list.file_path}")
        else:
            print("Failed to save list.")
    
    def _validate_date(self, date_str):
        """Validate that a string is in YYYY-MM-DD format."""
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False


if __name__ == "__main__":
    app = TodoApp()
    app.run()