"""
Format conversion module for ChatML, Alpaca, and JSONL formats.

This module provides comprehensive format conversion capabilities including:
- ChatML format for conversational models
- Alpaca format for instruction tuning
- JSONL format for general training
- Custom format definitions
"""

import json
import re
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod


class OutputFormat(Enum):
    """Supported output formats."""
    CHATML = "chatml"
    ALPACA = "alpaca"
    JSONL = "jsonl"
    CUSTOM = "custom"


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    role: str  # system, user, assistant
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InstructionExample:
    """Single instruction-following example."""
    instruction: str
    output: str
    input: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FormatConversionResult:
    """Result of format conversion operation."""
    formatted_data: List[str]
    format_type: OutputFormat
    total_examples: int
    conversion_errors: List[str]
    metadata: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate conversion success rate."""
        if self.total_examples == 0:
            return 1.0
        return (self.total_examples - len(self.conversion_errors)) / self.total_examples


class BaseFormatter(ABC):
    """Abstract base class for format converters."""
    
    @abstractmethod
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format a conversation into the target format."""
        pass
    
    @abstractmethod
    def format_instruction(self, example: InstructionExample) -> str:
        """Format an instruction example into the target format."""
        pass
    
    @abstractmethod
    def validate_format(self, formatted_text: str) -> bool:
        """Validate that the formatted text is correct."""
        pass


class ChatMLFormatter(BaseFormatter):
    """
    ChatML format converter.
    
    ChatML uses special tokens to denote conversation turns:
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>user
    Hello!
    <|im_end|>
    <|im_start|>assistant
    Hi there! How can I help you?
    <|im_end|>
    """
    
    def __init__(self, 
                 start_token: str = "<|im_start|>",
                 end_token: str = "<|im_end|>",
                 include_system: bool = True):
        """
        Initialize ChatML formatter.
        
        Args:
            start_token: Token to start each turn
            end_token: Token to end each turn
            include_system: Whether to include system messages
        """
        self.start_token = start_token
        self.end_token = end_token
        self.include_system = include_system
    
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format conversation turns into ChatML format."""
        formatted_turns = []
        
        for turn in turns:
            # Skip system messages if not including them
            if not self.include_system and turn.role == "system":
                continue
            
            formatted_turn = f"{self.start_token}{turn.role}\n{turn.content}\n{self.end_token}"
            formatted_turns.append(formatted_turn)
        
        return "\n".join(formatted_turns)
    
    def format_instruction(self, example: InstructionExample) -> str:
        """Format instruction example into ChatML conversation format."""
        turns = []
        
        # Add system message if available in metadata
        if example.metadata and "system" in example.metadata:
            turns.append(ConversationTurn(
                role="system",
                content=example.metadata["system"]
            ))
        
        # Create user turn with instruction and input
        user_content = example.instruction
        if example.input:
            user_content += f"\n\nInput: {example.input}"
        
        turns.append(ConversationTurn(role="user", content=user_content))
        turns.append(ConversationTurn(role="assistant", content=example.output))
        
        return self.format_conversation(turns)
    
    def validate_format(self, formatted_text: str) -> bool:
        """Validate ChatML format."""
        # Check for balanced start/end tokens
        start_count = formatted_text.count(self.start_token)
        end_count = formatted_text.count(self.end_token)
        
        if start_count != end_count:
            return False
        
        # Check for valid role names after start tokens
        pattern = rf"{re.escape(self.start_token)}(system|user|assistant)\n"
        matches = re.findall(pattern, formatted_text)
        
        return len(matches) == start_count


class AlpacaFormatter(BaseFormatter):
    """
    Alpaca format converter.
    
    Alpaca format structure:
    {
        "instruction": "...",
        "input": "...",  # optional
        "output": "..."
    }
    """
    
    def __init__(self, include_metadata: bool = False):
        """
        Initialize Alpaca formatter.
        
        Args:
            include_metadata: Whether to include metadata fields
        """
        self.include_metadata = include_metadata
    
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Convert conversation to Alpaca instruction format."""
        # Extract system message as context
        system_message = None
        user_turns = []
        assistant_turns = []
        
        for turn in turns:
            if turn.role == "system":
                system_message = turn.content
            elif turn.role == "user":
                user_turns.append(turn.content)
            elif turn.role == "assistant":
                assistant_turns.append(turn.content)
        
        # Create instruction from conversation
        if not user_turns or not assistant_turns:
            return ""
        
        instruction = "Continue this conversation appropriately."
        if system_message:
            instruction = f"{system_message}\n\n{instruction}"
        
        # Combine user turns as input
        input_text = "\n".join(user_turns)
        
        # Use first assistant response as output
        output_text = assistant_turns[0]
        
        example = InstructionExample(
            instruction=instruction,
            input=input_text,
            output=output_text
        )
        
        return self.format_instruction(example)
    
    def format_instruction(self, example: InstructionExample) -> str:
        """Format instruction example into Alpaca JSON format."""
        alpaca_dict = {
            "instruction": example.instruction,
            "output": example.output
        }
        
        # Add input if present
        if example.input:
            alpaca_dict["input"] = example.input
        
        # Add metadata if requested
        if self.include_metadata and example.metadata:
            alpaca_dict.update(example.metadata)
        
        return json.dumps(alpaca_dict, ensure_ascii=False)
    
    def validate_format(self, formatted_text: str) -> bool:
        """Validate Alpaca JSON format."""
        try:
            data = json.loads(formatted_text)
            
            # Check required fields
            if "instruction" not in data or "output" not in data:
                return False
            
            # Check field types
            if not isinstance(data["instruction"], str) or not isinstance(data["output"], str):
                return False
            
            # Check optional input field
            if "input" in data and not isinstance(data["input"], str):
                return False
            
            return True
        except json.JSONDecodeError:
            return False


class JSONLFormatter(BaseFormatter):
    """
    JSONL (JSON Lines) format converter.
    
    Each line is a separate JSON object with flexible structure.
    """
    
    def __init__(self, 
                 conversation_key: str = "messages",
                 instruction_key: str = "text",
                 include_metadata: bool = True):
        """
        Initialize JSONL formatter.
        
        Args:
            conversation_key: Key name for conversation data
            instruction_key: Key name for instruction data
            include_metadata: Whether to include metadata
        """
        self.conversation_key = conversation_key
        self.instruction_key = instruction_key
        self.include_metadata = include_metadata
    
    def format_conversation(self, turns: List[ConversationTurn]) -> str:
        """Format conversation into JSONL format."""
        messages = []
        for turn in turns:
            message = {
                "role": turn.role,
                "content": turn.content
            }
            if self.include_metadata and turn.metadata:
                message.update(turn.metadata)
            messages.append(message)
        
        jsonl_dict = {self.conversation_key: messages}
        return json.dumps(jsonl_dict, ensure_ascii=False)
    
    def format_instruction(self, example: InstructionExample) -> str:
        """Format instruction example into JSONL format."""
        # Create formatted text
        text_parts = [f"Instruction: {example.instruction}"]
        
        if example.input:
            text_parts.append(f"Input: {example.input}")
        
        text_parts.append(f"Output: {example.output}")
        
        jsonl_dict = {
            self.instruction_key: "\n\n".join(text_parts),
            "instruction": example.instruction,
            "output": example.output
        }
        
        if example.input:
            jsonl_dict["input"] = example.input
        
        if self.include_metadata and example.metadata:
            jsonl_dict.update(example.metadata)
        
        return json.dumps(jsonl_dict, ensure_ascii=False)
    
    def validate_format(self, formatted_text: str) -> bool:
        """Validate JSONL format."""
        try:
            json.loads(formatted_text)
            return True
        except json.JSONDecodeError:
            return False


class FormatConverter:
    """
    Main format converter class that handles multiple output formats.
    """
    
    def __init__(self):
        """Initialize format converter with all supported formatters."""
        self.formatters = {
            OutputFormat.CHATML: ChatMLFormatter(),
            OutputFormat.ALPACA: AlpacaFormatter(),
            OutputFormat.JSONL: JSONLFormatter()
        }
    
    def convert_conversations(self, 
                            conversations: List[List[ConversationTurn]],
                            output_format: OutputFormat) -> FormatConversionResult:
        """
        Convert multiple conversations to specified format.
        
        Args:
            conversations: List of conversation turn lists
            output_format: Target output format
            
        Returns:
            FormatConversionResult with converted data
        """
        if output_format not in self.formatters:
            raise ValueError(f"Unsupported format: {output_format}")
        
        formatter = self.formatters[output_format]
        formatted_data = []
        conversion_errors = []
        
        for i, conversation in enumerate(conversations):
            try:
                formatted = formatter.format_conversation(conversation)
                if formatter.validate_format(formatted):
                    formatted_data.append(formatted)
                else:
                    conversion_errors.append(f"Validation failed for conversation {i}")
            except Exception as e:
                conversion_errors.append(f"Error converting conversation {i}: {str(e)}")
        
        return FormatConversionResult(
            formatted_data=formatted_data,
            format_type=output_format,
            total_examples=len(conversations),
            conversion_errors=conversion_errors,
            metadata={
                "format": output_format.value,
                "formatter_type": type(formatter).__name__
            }
        )
    
    def convert_instructions(self,
                           instructions: List[InstructionExample],
                           output_format: OutputFormat) -> FormatConversionResult:
        """
        Convert instruction examples to specified format.
        
        Args:
            instructions: List of instruction examples
            output_format: Target output format
            
        Returns:
            FormatConversionResult with converted data
        """
        if output_format not in self.formatters:
            raise ValueError(f"Unsupported format: {output_format}")
        
        formatter = self.formatters[output_format]
        formatted_data = []
        conversion_errors = []
        
        for i, instruction in enumerate(instructions):
            try:
                formatted = formatter.format_instruction(instruction)
                if formatter.validate_format(formatted):
                    formatted_data.append(formatted)
                else:
                    conversion_errors.append(f"Validation failed for instruction {i}")
            except Exception as e:
                conversion_errors.append(f"Error converting instruction {i}: {str(e)}")
        
        return FormatConversionResult(
            formatted_data=formatted_data,
            format_type=output_format,
            total_examples=len(instructions),
            conversion_errors=conversion_errors,
            metadata={
                "format": output_format.value,
                "formatter_type": type(formatter).__name__
            }
        )
    
    def save_to_file(self, 
                     result: FormatConversionResult,
                     output_path: str,
                     append_mode: bool = False) -> None:
        """
        Save conversion result to file.
        
        Args:
            result: FormatConversionResult to save
            output_path: Path to output file
            append_mode: Whether to append to existing file
        """
        mode = 'a' if append_mode else 'w'
        
        with open(output_path, mode, encoding='utf-8') as f:
            for formatted_item in result.formatted_data:
                f.write(formatted_item + '\n')
    
    def add_custom_formatter(self, 
                           format_name: str,
                           formatter: BaseFormatter) -> None:
        """
        Add a custom formatter.
        
        Args:
            format_name: Name for the custom format
            formatter: Custom formatter instance
        """
        custom_format = OutputFormat.CUSTOM
        self.formatters[custom_format] = formatter
    
    def batch_convert_mixed(self,
                          conversations: List[List[ConversationTurn]],
                          instructions: List[InstructionExample],
                          output_format: OutputFormat) -> FormatConversionResult:
        """
        Convert mixed conversation and instruction data.
        
        Args:
            conversations: List of conversations
            instructions: List of instructions
            output_format: Target format
            
        Returns:
            Combined FormatConversionResult
        """
        # Convert conversations
        conv_result = self.convert_conversations(conversations, output_format)
        
        # Convert instructions
        inst_result = self.convert_instructions(instructions, output_format)
        
        # Combine results
        combined_data = conv_result.formatted_data + inst_result.formatted_data
        combined_errors = conv_result.conversion_errors + inst_result.conversion_errors
        total_examples = conv_result.total_examples + inst_result.total_examples
        
        return FormatConversionResult(
            formatted_data=combined_data,
            format_type=output_format,
            total_examples=total_examples,
            conversion_errors=combined_errors,
            metadata={
                "format": output_format.value,
                "conversations_count": len(conversations),
                "instructions_count": len(instructions),
                "mixed_conversion": True
            }
        )


def convert_to_chatml(conversations: List[List[ConversationTurn]]) -> List[str]:
    """
    Convenience function to convert conversations to ChatML format.
    
    Args:
        conversations: List of conversation turn lists
        
    Returns:
        List of ChatML formatted strings
    """
    converter = FormatConverter()
    result = converter.convert_conversations(conversations, OutputFormat.CHATML)
    return result.formatted_data


def convert_to_alpaca(instructions: List[InstructionExample]) -> List[str]:
    """
    Convenience function to convert instructions to Alpaca format.
    
    Args:
        instructions: List of instruction examples
        
    Returns:
        List of Alpaca formatted JSON strings
    """
    converter = FormatConverter()
    result = converter.convert_instructions(instructions, OutputFormat.ALPACA)
    return result.formatted_data


def convert_to_jsonl(data: Union[List[List[ConversationTurn]], List[InstructionExample]]) -> List[str]:
    """
    Convenience function to convert data to JSONL format.
    
    Args:
        data: Either conversations or instructions
        
    Returns:
        List of JSONL formatted strings
    """
    converter = FormatConverter()
    
    # Detect data type
    if data and isinstance(data[0], list):
        # Conversations
        result = converter.convert_conversations(data, OutputFormat.JSONL)
    else:
        # Instructions
        result = converter.convert_instructions(data, OutputFormat.JSONL)
    
    return result.formatted_data